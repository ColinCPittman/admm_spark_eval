import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel

import breeze.linalg.{DenseVector, norm}
import breeze.numerics.sqrt
import breeze.optimize.{DiffFunction, LBFGS}

import java.io.PrintWriter
import scala.collection.JavaConverters._

/**
 * SPARK 2.4 COMPATIBLE VERSION OF SuADMM
 * ================================================
 * This version is identical to the working Spark 4.0 implementation
 * with only the necessary compatibility fixes for Spark 2.4/Scala 2.11
 * 
 * 1. DATA STRUCTURES & INTERFACES
 * ================================================
 */

// local state for each partition in ADMM
case class ADMMState(
    localWeightVector_w: DenseVector[Double],
    scaledDualVariable_u: DenseVector[Double]
)

// interface for different ADMM update strategies  
trait ADMMUpdater extends Serializable {

    // solve local w subproblem on partition
    def wUpdate(
        partitionData: Iterator[(Double, DenseVector[Double])],
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double],
        penaltyParameter_rho: Double
    ): ADMMState

    // update global z on driver
    def zUpdate(
        averageLocalWeights_w_avg: DenseVector[Double],
        averageDualVariable_u_avg: DenseVector[Double],
        penaltyParameter_rho: Double,
        numPartitions: Long
    ): DenseVector[Double]

    // update dual variable u on partition
    def uUpdate(
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double]
    ): ADMMState

    // check convergence based on residuals
    def checkConvergence(
        primalResidual: Double,
        dualResidual: Double,
        normOfAllLocalWeights: Double,
        normOfGlobalZ: Double,
        numPartitions: Long,
        numFeatures: Int
    ): Boolean

    // adaptive rho update scheme
    def updateRho(
        currentRho: Double,
        primalResidual: Double,
        dualResidual: Double
    ): Double
}


/**
 *
 * 2. L2-REGULARIZED LOGISTIC REGRESSION
 * ============================================================================================
 * Implements the ADMMUpdater trait for L2-Regularized Logistic Regression (SquaredL2Updater in the diagram).
 * This class contains the specific mathematical formulas for the problem.
 *
 * @param regularizationParam_lambda The L2 regularization parameter, λ.
 */
class SquaredL2Updater(val regularizationParam_lambda: Double) extends ADMMUpdater {

        override def wUpdate(
        partitionData: Iterator[(Double, DenseVector[Double])],
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double],
        penaltyParameter_rho: Double
    ): ADMMState = {

        // convert iterator to array for multiple passes
        val dataBuffer = scala.collection.mutable.ArrayBuffer[(Double, DenseVector[Double])]()
        val maxSamples = 20000  // limited to make runnable with our resources
        var count = 0
        
        while (partitionData.hasNext && count < maxSamples) {
            dataBuffer += partitionData.next()
            count += 1
            
            // force GC every 1000 samples
            if (count % 1000 == 0) {
                System.gc()
            }
        }
        
        if (dataBuffer.isEmpty) return currentState
        println(s"processing ${dataBuffer.size} samples in partition")

        val costFunction = new DiffFunction[DenseVector[Double]] {
            def calculate(w: DenseVector[Double]): (Double, DenseVector[Double]) = {
                var totalLoss = 0.0
                val gradient = DenseVector.zeros[Double](w.length)
                val miniSize = 100
                
                // process in batches
                for (start <- dataBuffer.indices by miniSize) {
                    val end = math.min(start + miniSize, dataBuffer.size)
                    var batchLoss = 0.0
                    
                    // accumulate gradient in-place to which should avoid creating intermediate vectors
                    for (i <- start until end) {
                        val (label, features) = dataBuffer(i)
                        val prediction = features.t * w
                        val expTerm = math.exp(-label * prediction)
                        
                        // logistic loss
                        batchLoss += math.log(1 + expTerm)
                        
                        // accumulate directly into gradient vector
                        val gradientCoeff = -label / (1 + math.exp(label * prediction))
                        var j = 0
                        while (j < features.length) {
                            gradient(j) += features(j) * gradientCoeff
                            j += 1
                        }
                    }
                    
                    totalLoss += batchLoss
                    
                    // force GC every few mini-batches
                    if (start % (miniSize * 5) == 0) {
                        System.gc()
                    }
                }
                
                // normalize by number of samples
                totalLoss = totalLoss / dataBuffer.size
                gradient :/= dataBuffer.size.toDouble

                // augmented Lagrangian term
                val lagrangianResidual = w - globalConsensusVariable_z + currentState.scaledDualVariable_u
                val augmentedLagrangian = (penaltyParameter_rho / 2.0) * (lagrangianResidual.t * lagrangianResidual)
                val lagrangianGradient = lagrangianResidual * penaltyParameter_rho

                // combining objective and gradient
                (totalLoss + augmentedLagrangian, gradient + lagrangianGradient)
            }
        }

        val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 50, m = 5, tolerance = 1e-4)
        val new_w = lbfgs.minimize(costFunction, currentState.localWeightVector_w)

        currentState.copy(localWeightVector_w = new_w)
    }

    // z-update on driver: z = (N*ρ / (N*ρ + 2*λ)) * (w_avg + u_avg)
    override def zUpdate(
        averageLocalWeights_w_avg: DenseVector[Double],
        averageDualVariable_u_avg: DenseVector[Double],
        penaltyParameter_rho: Double,
        numPartitions: Long
    ): DenseVector[Double] = {
        val N = numPartitions.toDouble
        val numerator = N * penaltyParameter_rho
        val denominator = numerator + (2 * regularizationParam_lambda)
        val shrinkageFactor = numerator / denominator

        (averageLocalWeights_w_avg + averageDualVariable_u_avg) * shrinkageFactor
    }

    // u-update: u = u + w - z
    override def uUpdate(
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double]
    ): ADMMState = {
        val new_u = currentState.scaledDualVariable_u + currentState.localWeightVector_w - globalConsensusVariable_z
        currentState.copy(scaledDualVariable_u = new_u)
    }

    // convergence check based on primal residual
    override def checkConvergence(
        primalResidual: Double,
        dualResidual: Double,
        normOfAllLocalWeights: Double,
        normOfGlobalZ: Double,
        numPartitions: Long,
        numFeatures: Int
    ): Boolean = {
        val epsilon_absolute = 1e-3
        val epsilon_relative = 1e-3
        val N = numPartitions.toDouble
        
        // primal tolerance from paper
        val primalTolerance = math.sqrt(N) * epsilon_absolute + epsilon_relative * math.max(normOfAllLocalWeights, normOfGlobalZ)

        println(f"Primal Residual: $primalResidual%.6f, Primal Tolerance: $primalTolerance%.6f")
        println(f"Dual Residual:   $dualResidual%.6f (for monitoring only)")

        primalResidual <= primalTolerance
    }

    // adaptive rho update
    override def updateRho(
        currentRho: Double,
        primalResidual: Double,
        dualResidual: Double
    ): Double = {
        val mu = 10.0
        val tau_incr = 2.0
        val tau_decr = 2.0

        if (primalResidual > mu * dualResidual) {
            currentRho * tau_incr
        } else if (dualResidual > mu * primalResidual) {
            currentRho / tau_decr
        } else {
            currentRho
        }
    }
}


/**
 * 3. EXECUTION LOGIC
 * ================================================
 */

// main ADMM optimizer that coordinates the distributed process
class ADMMOptimizer(
    val updater: ADMMUpdater,
    val numPartitions: Int,
    val numFeatures: Int) extends Serializable {

    def run(data: RDD[(Double, DenseVector[Double])], maxIterations: Int): (DenseVector[Double], Int) = {
        val sparkContext = data.sparkContext

        val partitionedData = data.repartition(numPartitions).persist(StorageLevel.DISK_ONLY)
        val numActualPartitions = partitionedData.getNumPartitions

        // Initialize ADMMState on each partition with random vectors for w and zero for u.
        var statesRDD = partitionedData.mapPartitions(_ => Iterator(ADMMState(
            localWeightVector_w = DenseVector.rand[Double](numFeatures) * 0.01,
            scaledDualVariable_u = DenseVector.zeros[Double](numFeatures)
        )), preservesPartitioning = true).persist(StorageLevel.DISK_ONLY)

        // Initialize global variables on the driver
        var globalConsensusVariable_z = DenseVector.zeros[Double](numFeatures)
        var previousGlobalZ = DenseVector.zeros[Double](numFeatures)
        var penaltyParameter_rho = 1.0

        var isConverged = false
        var iteration = 0
        // Enforce a minimum number of iterations before we allow convergence.
        val minIterationsRequired = 10

        while (!isConverged && iteration < maxIterations) {
            iteration += 1
            println(s"\n--- Iteration: $iteration ---")

            // Broadcast global variables to all executors
            val z_broadcast = sparkContext.broadcast(globalConsensusVariable_z)
            val rho_broadcast = sparkContext.broadcast(penaltyParameter_rho)

            // Combine data and state RDDs for processing
            val dataAndStatesRDD = partitionedData.zipPartitions(statesRDD, preservesPartitioning = true) {
                (dataIter, stateIter) => Iterator((dataIter, stateIter.next))
            }

            // Step 1: w-update (in parallel on each partition)
            val states_after_w_update = dataAndStatesRDD.map { case (partitionData, state) =>
                updater.wUpdate(partitionData, state, z_broadcast.value, rho_broadcast.value)
            }.persist(StorageLevel.DISK_ONLY)

            // Step 2: Aggregate results for z-update
            // We use treeAggregate for efficiency, as recommended in the paper.
            // We compute sum(w), sum(u), and sum(||w||²) in a single pass.
            val (w_sum, u_sum, w_squared_norm_sum) = states_after_w_update
              .map(s => (s.localWeightVector_w, s.scaledDualVariable_u))
              .treeAggregate(
                (DenseVector.zeros[Double](numFeatures), DenseVector.zeros[Double](numFeatures), 0.0) // Zero value
              )(
                seqOp = (agg, wu_tuple) => {
                    val (w, u) = wu_tuple
                    val w_norm_sq = norm(w, 2) * norm(w, 2)
                    (agg._1 + w, agg._2 + u, agg._3 + w_norm_sq)
                },
                combOp = (agg1, agg2) => {
                    (agg1._1 + agg2._1, agg1._2 + agg2._2, agg1._3 + agg2._3)
                }
              )

            val w_avg = w_sum / numActualPartitions.toDouble
            val u_avg = u_sum / numActualPartitions.toDouble

            // Step 3: z-update (on driver)
            previousGlobalZ = globalConsensusVariable_z
            globalConsensusVariable_z = updater.zUpdate(w_avg, u_avg, penaltyParameter_rho, numActualPartitions)
            val new_z_broadcast = sparkContext.broadcast(globalConsensusVariable_z)

            // Step 4: u-update (in parallel on each partition)
            val states_after_u_update = states_after_w_update.map { state =>
                updater.uUpdate(state, new_z_broadcast.value)
            }.persist(StorageLevel.DISK_ONLY)

            // Step 5: Check for convergence
            // Primal residual r^k = sqrt( Σ ||w_i - z||² )
            val primalResidual_squared_sum = states_after_u_update.map(s => norm(s.localWeightVector_w - globalConsensusVariable_z, 2)).map(n => n * n).sum()
            val primalResidual = math.sqrt(primalResidual_squared_sum)
            
            // Dual residual s^k = ρ * sqrt(N) * ||z^k - z^(k-1)||  (consensus form)
            val dualResidual = penaltyParameter_rho * math.sqrt(numActualPartitions) * norm(globalConsensusVariable_z - previousGlobalZ, 2)

            val normOfAllLocalWeights = math.sqrt(w_squared_norm_sum)
            val normOfGlobalZ = norm(globalConsensusVariable_z, 2)

            isConverged = updater.checkConvergence(primalResidual, dualResidual, normOfAllLocalWeights, normOfGlobalZ, numActualPartitions, numFeatures)

            // Prevent premature convergence in very early iterations
            if (iteration < minIterationsRequired) {
                isConverged = false
            }

            // Step 6: Update rho for the next iteration
            penaltyParameter_rho = updater.updateRho(penaltyParameter_rho, primalResidual, dualResidual)
            println(s"New Rho: $penaltyParameter_rho")

            // Cleanup for next iteration
            statesRDD.unpersist()
            states_after_w_update.unpersist()
            statesRDD = states_after_u_update // The new state for the next iteration
        }

        partitionedData.unpersist()
        statesRDD.unpersist()

        println(s"\nConvergence reached after $iteration iterations.")
        (globalConsensusVariable_z, iteration) // The final model and iteration count
    }
}

/**
 * A user-facing class that wraps the ADMM optimizer for logistic regression.
 * This corresponds to the `SparseLogisticRegressionWithADMM` class in the UML diagram,
 * but is named more generically as the implementation uses dense vectors.
 * It inherits from a placeholder `GeneralizedLinearAlgorithm` to match the diagram.
 */
abstract class GeneralizedLinearAlgorithm
class LogisticRegressionWithADMM(
    val regularizationParam_lambda: Double,
    val numPartitions: Int,
    val maxIterations: Int
) extends GeneralizedLinearAlgorithm {

    def train(data: RDD[(Double, DenseVector[Double])]): (DenseVector[Double], Int) = {
        require(data.getStorageLevel != StorageLevel.NONE, "Input RDD must be persisted.")
        val numFeatures = data.first()._2.length
        
        println(s"Running ADMM with $numPartitions partitions, $numFeatures features.")
        println(s"Lambda: $regularizationParam_lambda, Max Iterations: $maxIterations")

        val updater = new SquaredL2Updater(regularizationParam_lambda)
        val optimizer = new ADMMOptimizer(updater, numPartitions, numFeatures)
        
        optimizer.run(data, maxIterations)
    }
}


/**
 * 4. RUNNER OBJECT
 * ============================================================================================
 */

/**
 * A runnable object with a main method to load data, execute the optimizer, and save the results.
 */
object ADMMRunner {
    def main(args: Array[String]): Unit = {
        if (args.length < 4) {
            System.err.println("Usage: ADMMRunner <data_path> <num_partitions> <num_features> <output_path>")
            System.exit(1)
        }

        val dataPath = args(0)
        val numPartitions = args(1).toInt
        val numFeatures = args(2).toInt
        val outputPath = args(3)
        
        // --- Hyperparameters ---
        val lambda = 0.1
        val maxIterations = 50

        val spark = SparkSession.builder.appName("ADMM Logistic Regression").getOrCreate()
        val sc = spark.sparkContext

        println(s"Loading data from: $dataPath")
        val data = MLUtils.loadLibSVMFile(sc, dataPath).map {
            labeledPoint => (labeledPoint.label, DenseVector(labeledPoint.features.toArray))
        }.persist(StorageLevel.DISK_ONLY)
        
        // Instantiate the user-facing algorithm class
        val admmLogisticRegression = new LogisticRegressionWithADMM(
            regularizationParam_lambda = lambda,
            numPartitions = numPartitions,
            maxIterations = maxIterations
        )

        // Train the model
        val (trainedModel, iterations) = admmLogisticRegression.train(data)
        
        // Save the model
        new PrintWriter(outputPath) {
            write(trainedModel.toArray.mkString(","))
            close()
        }
        
        println(s"Training completed in $iterations iterations")
        println(s"Model saved to: $outputPath")
        
        spark.stop()
    }
}

/**
 * 5. CONVENIENCE FUNCTIONS FOR SPARK SHELL (SPARK 2.4 COMPATIBLE)
 * ============================================================================================
 */

// custom data loading that handles both 0-based and 1-based feature indices
def loadLibSVMFileCustom(sc: SparkContext, path: String): RDD[org.apache.spark.mllib.regression.LabeledPoint] = {
  sc.textFile(path).map { line =>
    val parts = line.trim.split(" ")
    val label = parts(0).toDouble
    
    // parse features, handling both 0-based and 1-based indices
    val features = parts.tail.map { feature =>
      val Array(indexStr, valueStr) = feature.split(":")
      val index = indexStr.toInt
      val value = valueStr.toDouble
      // convert 0-based to 1-based if needed
      (if (index == 0) 1 else if (index < 0) index + 1 else index, value)
    }.sortBy(_._1)
    
    // find max feature index to determine vector size
    val maxIndex = if (features.isEmpty) 1 else features.map(_._1).max
    val featureArray = Array.fill(maxIndex)(0.0)
    
    features.foreach { case (index, value) =>
      featureArray(index - 1) = value  // convert back to 0-based for array
    }
    
    new org.apache.spark.mllib.regression.LabeledPoint(label, org.apache.spark.mllib.linalg.Vectors.dense(featureArray))
  }
}

// smart data loading that uses custom parser for HIGGS and standard for others
def loadDataSmart(sc: SparkContext, path: String, isHiggs: Boolean): RDD[org.apache.spark.mllib.regression.LabeledPoint] = {
  if (isHiggs) {
    println("Using custom parser for HIGGS dataset (0-based indices)")
    loadLibSVMFileCustom(sc, path)
  } else {
    MLUtils.loadLibSVMFile(sc, path)
  }
}

/**
 * 6. ACCURACY EVALUATION FUNCTIONS (KEY FOR PAPER REPRODUCTION)
 * ============================================================================================
 */

/**
 * Calculate logistic regression prediction accuracy.
 * This is essential for reproducing the paper's claimed 98.17% accuracy.
 */
def calculateAccuracy(testData: RDD[(Double, DenseVector[Double])], model: DenseVector[Double]): Double = {
  val predictions = testData.map { case (actualLabel, features) =>
    // Logistic regression prediction: sigmoid(x^T * w)
    val rawPrediction = features.t * model
    val probability = 1.0 / (1.0 + math.exp(-rawPrediction))
    val predictedLabel = if (probability > 0.5) 1.0 else -1.0
    
    (actualLabel, predictedLabel)
  }
  
  val correct = predictions.filter { case (actual, predicted) => actual == predicted }.count()
  val total = predictions.count()
  
  correct.toDouble / total.toDouble
}


def splitTrainTest(data: RDD[(Double, DenseVector[Double])], trainRatio: Double = 0.8): (RDD[(Double, DenseVector[Double])], RDD[(Double, DenseVector[Double])]) = {
  val Array(trainData, testData) = data.randomSplit(Array(trainRatio, 1.0 - trainRatio), seed = 42)
  (trainData.persist(StorageLevel.MEMORY_AND_DISK), testData.persist(StorageLevel.MEMORY_AND_DISK))
}



def runADMM(
    dataset: String = "rcv1",
    trainFilename: String = "",
    testFilename: String = "",
    numPartitions: Int = 8,
    lambda: Double = 0.01,
    maxIterations: Int = 50): DenseVector[Double] = {

  val (defaultTrain, defaultTest) = dataset.toLowerCase match {
    case "higgs" => ("higgs_train15k.binary", "higgs_test10k.binary")
    case "rcv1" | _ => ("rcv1_train15k.binary", "rcv1_test10k.binary")
  }
  
  val finalTrainFile = if (trainFilename.nonEmpty) trainFilename else defaultTrain
  val finalTestFile = if (testFilename.nonEmpty) testFilename else defaultTest

  val trainPath = if (finalTrainFile.startsWith("/")) finalTrainFile else s"/workspace/data/sourced/$finalTrainFile"
  val testPath = if (finalTestFile.startsWith("/")) finalTestFile else s"/workspace/data/sourced/$finalTestFile"
  
  val output = new StringBuilder()
  
  output.append(s"Dataset: ${dataset.toUpperCase()}\n")
  output.append(s"Loading training data from: $trainPath\n")
  output.append(s"Loading test data from: $testPath\n")
  output.append("Using optimized samples with automatic rho updating (μ=10, τ^incr=2, τ^decr=2)\n")
  
  println(s"Dataset: ${dataset.toUpperCase()}")
  println(s"Loading training data from: $trainPath")
  println(s"Loading test data from: $testPath")
  println("Using optimized samples with automatic rho updating (μ=10, τ^incr=2, τ^decr=2)")

  System.setProperty("spark.ui.showConsoleProgress", "false")
  org.apache.log4j.Logger.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(org.apache.log4j.Level.ERROR)

  val trainData = loadDataSmart(sc, trainPath, dataset.toLowerCase == "higgs")
    .map { lp => (lp.label, DenseVector(lp.features.toArray)) }
    .repartition(numPartitions)
    .persist(StorageLevel.DISK_ONLY)

  val testData = loadDataSmart(sc, testPath, dataset.toLowerCase == "higgs")
    .map { lp => (lp.label, DenseVector(lp.features.toArray)) }
    .persist(StorageLevel.DISK_ONLY)

  val nTrainSamples = trainData.count()
  val nTestSamples = testData.count()
  val nFeatures = trainData.first()._2.length
  
  output.append(s"Training: $nTrainSamples samples, Testing: $nTestSamples samples, Features: $nFeatures\n")
  println(s"Training: $nTrainSamples samples, Testing: $nTestSamples samples, Features: $nFeatures")

  val labelCounts = trainData.map(_._1).countByValue()
  output.append(s"Training label distribution: $labelCounts\n\n")
  println(s"Training label distribution: $labelCounts")

  val learner = new LogisticRegressionWithADMM(lambda, numPartitions, maxIterations)
  
  output.append("="*60 + "\n")
  output.append("STARTING ADMM TRAINING\n")
  output.append("="*60 + "\n")
  
  println("\n" + "="*60)
  println("STARTING ADMM TRAINING")
  println("="*60)

  val t0 = System.currentTimeMillis()
  val (model, iterations) = learner.train(trainData)
  val t1 = System.currentTimeMillis()
  val runtimeSec = (t1 - t0) / 1000.0

 
  val accuracy = calculateAccuracy(testData, model) * 100.0

  val modelNorm = norm(model, 2)
  val nonZeroWeights = model.toArray.count(math.abs(_) > 1e-6)
  

  val samplePredictions = testData.take(5).map { case (actualLabel, features) =>
    val rawPrediction = features.t * model
    val probability = 1.0 / (1.0 + math.exp(-rawPrediction))
    val predictedLabel = if (probability > 0.5) 1.0 else -1.0
    (actualLabel, rawPrediction, probability, predictedLabel)
  }

  output.append("\n" + "="*60 + "\n")
  output.append("ADMM RESULTS\n")
  output.append("="*60 + "\n")
  output.append(s"Training Dataset: $finalTrainFile ($nTrainSamples samples)\n")
  output.append(s"Test Dataset: $finalTestFile ($nTestSamples samples)\n")
  output.append(s"Features: $nFeatures\n")
  output.append(s"Runtime: ${f"$runtimeSec%.1f"} seconds\n")
  output.append(s"Accuracy: ${f"$accuracy%.2f"}%\n")
  output.append(s"Iterations: $iterations\n")
  output.append(s"Parameters: λ=$lambda, Partitions=$numPartitions\n")
  output.append(s"Model L2 norm: ${f"$modelNorm%.6f"}, Non-zero weights: $nonZeroWeights/${model.length}\n")
  output.append("="*60 + "\n\n")
  output.append("MODEL WEIGHTS:\n")
  output.append(model.toArray.mkString(",") + "\n")

  println("\n" + "="*60)
  println("ADMM RESULTS")
  println("="*60)
  println(s"Training Dataset: $finalTrainFile ($nTrainSamples samples)")
  println(s"Test Dataset: $finalTestFile ($nTestSamples samples)")
  println(s"Features: $nFeatures")
  println(s"Runtime: ${f"$runtimeSec%.1f"} seconds")
  println(s"Accuracy: ${f"$accuracy%.2f"}%")
  println(s"Iterations: $iterations")
  println(s"Parameters: λ=$lambda, Partitions=$numPartitions")
  println(s"Model L2 norm: ${f"$modelNorm%.6f"}, Non-zero weights: $nonZeroWeights/${model.length}")
  println("="*60)


  val sparkVersion = "2_4"  
  

  val trainSamples = finalTrainFile.split("/").last.replaceAll(".*train([0-9]+k?).*", "$1")
  val testSamples = finalTestFile.split("/").last.replaceAll(".*test([0-9]+k?).*", "$1")
  

  def getNextRunNumber(basePattern: String): Int = {
    val outputDir = new java.io.File("/workspace/data/generated")
    if (!outputDir.exists()) outputDir.mkdirs()
    
    val existingFiles = outputDir.listFiles()
      .filter(_.getName.matches(s"${basePattern}_\\d+_${numPartitions}part_output\\.txt"))
      .map(_.getName.replaceAll(s"${basePattern}_(\\d+)_${numPartitions}part_output\\.txt", "$1").toInt)
    
    if (existingFiles.isEmpty) 1 else existingFiles.max + 1
  }
  
  val basePattern = s"${sparkVersion}_${trainSamples}_${testSamples}_admm"
  val runNumber = getNextRunNumber(basePattern)
  val out = s"/workspace/data/generated/${basePattern}_${runNumber}_${numPartitions}part_output.txt"
  
  new PrintWriter(out) { write(output.toString()); close() }
  println(s"Complete output saved to: $out")

  trainData.unpersist()
  testData.unpersist()

  model
}

def runLBFGS(
    dataset: String = "rcv1",
    trainFilename: String = "",
    testFilename: String = "",  
    numPartitions: Int = 8,
    regParam: Double = 0.01,
    maxIterations: Int = 100): org.apache.spark.mllib.linalg.Vector = {

  val (defaultTrain, defaultTest) = dataset.toLowerCase match {
    case "higgs" => ("higgs_train15k.binary", "higgs_test10k.binary")
    case "rcv1" | _ => ("rcv1_train15k.binary", "rcv1_test10k.binary")
  }
  
  val finalTrainFile = if (trainFilename.nonEmpty) trainFilename else defaultTrain
  val finalTestFile = if (testFilename.nonEmpty) testFilename else defaultTest

  val trainPath = if (finalTrainFile.startsWith("/")) finalTrainFile else s"/workspace/data/sourced/$finalTrainFile"
  val testPath = if (finalTestFile.startsWith("/")) finalTestFile else s"/workspace/data/sourced/$finalTestFile"
  
  val output = new StringBuilder()
  
  output.append(s"Dataset: ${dataset.toUpperCase()}\n")
  output.append(s"Loading training data from: $trainPath\n")
  output.append(s"Loading test data from: $testPath\n")
  output.append("Using MLlib LogisticRegressionWithLBFGS for baseline comparison\n")
  
  println(s"Dataset: ${dataset.toUpperCase()}")
  println(s"Loading training data from: $trainPath")
  println(s"Loading test data from: $testPath")
  println("Using MLlib LogisticRegressionWithLBFGS for baseline comparison")

  System.setProperty("spark.ui.showConsoleProgress", "false")
  org.apache.log4j.Logger.getLogger("org.apache.spark.scheduler.DAGScheduler").setLevel(org.apache.log4j.Level.ERROR)


  val trainData = loadDataSmart(sc, trainPath, dataset.toLowerCase == "higgs")
    .map(lp => lp.copy(label = if (lp.label == -1.0) 0.0 else 1.0))
    .repartition(numPartitions)
    .persist(StorageLevel.DISK_ONLY)

  val testData = loadDataSmart(sc, testPath, dataset.toLowerCase == "higgs")
    .map(lp => lp.copy(label = if (lp.label == -1.0) 0.0 else 1.0))
    .persist(StorageLevel.DISK_ONLY)

  val nTrainSamples = trainData.count()
  val nTestSamples = testData.count()
  val nFeatures = trainData.first().features.size
  
  output.append(s"Training: $nTrainSamples samples, Testing: $nTestSamples samples, Features: $nFeatures\n")
  println(s"Training: $nTrainSamples samples, Testing: $nTestSamples samples, Features: $nFeatures")


  val labelCounts = trainData.map(_.label).countByValue()
  output.append(s"Training label distribution: ${labelCounts}\n\n")
  println(s"Training label distribution: ${labelCounts}")


  val lbfgs = new org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS()
    .setNumClasses(2)
    .setIntercept(true)
  

  lbfgs.optimizer
    .setRegParam(regParam)
    .setNumIterations(maxIterations)
    .setConvergenceTol(1e-6)

  output.append("="*60 + "\n")
  output.append("STARTING LBFGS TRAINING\n")
  output.append("="*60 + "\n")
  
  println("\n" + "="*60)
  println("STARTING LBFGS TRAINING")
  println("="*60)
  
  val t0 = System.currentTimeMillis()
  val model = lbfgs.run(trainData)
  val t1 = System.currentTimeMillis()
  val runtimeSec = (t1 - t0) / 1000.0

  val predictions = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }
  
  val accuracy = predictions.filter { case (actual, predicted) => actual == predicted }.count().toDouble / nTestSamples * 100.0


  val modelWeights = model.weights.toArray
  val modelNorm = math.sqrt(modelWeights.map(x => x * x).sum)
  val nonZeroWeights = modelWeights.count(math.abs(_) > 1e-6)
  

  val samplePredictions = testData.take(5).map { point =>
    val predictedClass = model.predict(point.features)
    (point.label, predictedClass)
  }

  output.append("\n" + "="*60 + "\n")
  output.append("LBFGS RESULTS\n")
  output.append("="*60 + "\n")
  output.append(s"Training Dataset: $finalTrainFile ($nTrainSamples samples)\n")
  output.append(s"Test Dataset: $finalTestFile ($nTestSamples samples)\n")
  output.append(s"Features: $nFeatures\n")
  output.append(s"Runtime: ${f"$runtimeSec%.1f"} seconds\n")
  output.append(s"Accuracy: ${f"$accuracy%.2f"}%\n")
  output.append(s"Iterations: $maxIterations (fixed)\n")
  output.append(s"Parameters: regParam=$regParam, Partitions=$numPartitions\n")
  output.append(s"Model L2 norm: ${f"$modelNorm%.6f"}, Non-zero weights: $nonZeroWeights/${modelWeights.length}\n")
  output.append("="*60 + "\n\n")
  output.append("MODEL WEIGHTS:\n")
  output.append(modelWeights.mkString(",") + "\n")

  println("\n" + "="*60)
  println("LBFGS RESULTS")
  println("="*60)
  println(s"Training Dataset: $finalTrainFile ($nTrainSamples samples)")
  println(s"Test Dataset: $finalTestFile ($nTestSamples samples)")
  println(s"Features: $nFeatures")
  println(s"Runtime: ${f"$runtimeSec%.1f"} seconds")
  println(s"Accuracy: ${f"$accuracy%.2f"}%")
  println(s"Iterations: $maxIterations (fixed)")
  println(s"Parameters: regParam=$regParam, Partitions=$numPartitions")
  println(s"Model L2 norm: ${f"$modelNorm%.6f"}, Non-zero weights: $nonZeroWeights/${modelWeights.length}")
  println("="*60)


  val sparkVersion = "2_4"  
  

  val trainSamples = finalTrainFile.split("/").last.replaceAll(".*train([0-9]+k?).*", "$1")
  val testSamples = finalTestFile.split("/").last.replaceAll(".*test([0-9]+k?).*", "$1")
  

  def getNextRunNumber(basePattern: String): Int = {
    val outputDir = new java.io.File("/workspace/data/generated")
    if (!outputDir.exists()) outputDir.mkdirs()
    
    val existingFiles = outputDir.listFiles()
      .filter(_.getName.matches(s"${basePattern}_\\d+_${numPartitions}part_output\\.txt"))
      .map(_.getName.replaceAll(s"${basePattern}_(\\d+)_${numPartitions}part_output\\.txt", "$1").toInt)
    
    if (existingFiles.isEmpty) 1 else existingFiles.max + 1
  }
  
  val basePattern = s"${sparkVersion}_${trainSamples}_${testSamples}_lbfgs"
  val runNumber = getNextRunNumber(basePattern)
  val out = s"/workspace/data/generated/${basePattern}_${runNumber}_${numPartitions}part_output.txt"
  
  new PrintWriter(out) { write(output.toString()); close() }
  println(s"Complete output saved to: $out")


  trainData.unpersist()
  testData.unpersist()
  

  model.weights
}




