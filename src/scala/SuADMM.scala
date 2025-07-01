import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel

import breeze.linalg.{DenseVector, norm}
import breeze.numerics.sqrt
import breeze.optimize.{DiffFunction, LBFGS}

import java.io.PrintWriter

/**
 * ============================================================================================
 * 1. DATA STRUCTURES & INTERFACES (as per UML Diagram)
 * ============================================================================================
 */

/**
 * Represents the state for a single data partition in the ADMM algorithm.
 * As described in the paper, each partition `j` maintains its own local model `w_j` and
 * a scaled dual variable `u_j`.
 *
 * @param localWeightVector_w  The local model parameters `w` for this partition.
 * @param scaledDualVariable_u The scaled dual variable `u` for this partition.
 */
case class ADMMState(
    localWeightVector_w: DenseVector[Double],
    scaledDualVariable_u: DenseVector[Double]
)

/**
 * A trait (interface) for the ADMM update logic, as shown in the UML diagram.
 * This separates the core mathematical operations from the orchestration logic, allowing
 * different optimization problems (e.g., Lasso, SVM) to be implemented by creating new
 * classes that extend this trait.
 */
trait ADMMUpdater extends Serializable {

    /** Solves the subproblem for the local weight vector `w` on a single partition (Equation 3). */
    def wUpdate(
        partitionData: Iterator[(Double, DenseVector[Double])],
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double],
        penaltyParameter_rho: Double
    ): ADMMState

    /** Updates the global consensus variable `z` on the driver (Equation 4). */
    def zUpdate(
        averageLocalWeights_w_avg: DenseVector[Double],
        averageDualVariable_u_avg: DenseVector[Double],
        penaltyParameter_rho: Double,
        numPartitions: Long
    ): DenseVector[Double]

    /** Updates the scaled dual variable `u` on a single partition (Equation 5). */
    def uUpdate(
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double]
    ): ADMMState

    /** Determines if the algorithm has converged by checking residuals against tolerances (Equation 7). */
    def checkConvergence(
        primalResidual: Double,
        dualResidual: Double,
        normOfAllLocalWeights: Double,
        normOfGlobalZ: Double,
        numPartitions: Long,
        numFeatures: Int
    ): Boolean

    /** Implements the adaptive rho update scheme to improve convergence speed (Equation 8). */
    def updateRho(
        currentRho: Double,
        primalResidual: Double,
        dualResidual: Double
    ): Double
}


/**
 * ============================================================================================
 * 2. CONCRETE IMPLEMENTATION FOR L2-REGULARIZED LOGISTIC REGRESSION
 * ============================================================================================
 */

/**
 * Implements the ADMMUpdater trait for L2-Regularized Logistic Regression (SquaredL2Updater in the diagram).
 * This class contains the specific mathematical formulas for the problem.
 *
 * @param regularizationParam_lambda The L2 regularization parameter, λ.
 */
class SquaredL2Updater(val regularizationParam_lambda: Double) extends ADMMUpdater {

    /**
     * Performs the w-update on a single partition using L-BFGS.
     * This solves Equation (3) from the paper:
     *   argmin_w( (1/|S_j|) * Σ log(1+exp(-y*x_i'*w)) + (ρ/2) * ||w - z^k + u^k||² )
     * where S_j is the set of samples in the partition.
     */
    override def wUpdate(
        partitionData: Iterator[(Double, DenseVector[Double])],
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double],
        penaltyParameter_rho: Double
    ): ADMMState = {

        val data = partitionData.toList
        if (data.isEmpty) return currentState

        val costFunction = new DiffFunction[DenseVector[Double]] {
            def calculate(w: DenseVector[Double]): (Double, DenseVector[Double]) = {
                // Logistic loss term
                val loss = data.map { case (label, features) =>
                    math.log(1 + math.exp(-label * (features.t * w)))
                }.sum / data.size

                val lossGradient = data.map { case (label, features) =>
                    features * (-label / (1 + math.exp(label * (features.t * w))))
                }.reduce(_ + _) / data.size.toDouble

                // Augmented Lagrangian term
                val lagrangianResidual = w - globalConsensusVariable_z + currentState.scaledDualVariable_u
                val augmentedLagrangian = (penaltyParameter_rho / 2.0) * (lagrangianResidual.t * lagrangianResidual)
                val lagrangianGradient = lagrangianResidual * penaltyParameter_rho

                // Combine objective and gradient
                (loss + augmentedLagrangian, lossGradient + lagrangianGradient)
            }
        }

        // The L-BFGS solver minimizes the cost function.
        // As per the "Warm Start" section (3.5), we use the previous iteration's `w` as the starting point.
        val lbfgs = new LBFGS[DenseVector[Double]](100, 7, 1e-5)
        val new_w = lbfgs.minimize(costFunction, currentState.localWeightVector_w)

        currentState.copy(localWeightVector_w = new_w)
    }

    /**
     * Performs the z-update on the driver.
     * This solves Equation (4) from the paper:
     *   z^(k+1) = (N*ρ / (N*ρ + 2*λ)) * (w_avg^(k+1) + u_avg^k)
     */
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

    /**
     * Performs the u-update on each partition.
     * This performs Equation (5) from the paper:
     *   u^(k+1) = u^k + w^(k+1) - z^(k+1)
     */
    override def uUpdate(
        currentState: ADMMState,
        globalConsensusVariable_z: DenseVector[Double]
    ): ADMMState = {
        val new_u = currentState.scaledDualVariable_u + currentState.localWeightVector_w - globalConsensusVariable_z
        currentState.copy(scaledDualVariable_u = new_u)
    }

    /**
     * Checks for convergence based on primal and dual residuals.
     * Implements the stopping criterion from Equation (7) in the paper:
     *   ||r^k|| <= 𝜖𝑝𝑟𝑖 = sqrt(N*p)*𝜖_abs + 𝜖_rel * max{ ||w||, ||z|| }
     * where ||w|| is sqrt(Σ ||w_i||²).
     * The dual residual stopping condition is analogous.
     */
    override def checkConvergence(
        primalResidual: Double,
        dualResidual: Double,
        normOfAllLocalWeights: Double, // This is sqrt(Σ ||w_i||²)
        normOfGlobalZ: Double,
        numPartitions: Long,
        numFeatures: Int
    ): Boolean = {
        // Tolerances from paper Section 3.1
        val epsilon_absolute = 1e-3
        val epsilon_relative = 1e-3

        val N = numPartitions.toDouble
        val p = numFeatures.toDouble // p is dimension of features

        // Primal tolerance 𝜖𝑝𝑟𝑖 (from equation 7)
        val primalTolerance = sqrt(N * p) * epsilon_absolute + epsilon_relative * math.max(normOfAllLocalWeights, normOfGlobalZ)
        
        // Dual tolerance 𝜖𝑑𝑢𝑎𝑙 (analogous to primal)
        // The paper doesn't give an explicit dual tolerance formula; a common choice mirrors the primal one.
        val dualTolerance = sqrt(N * p) * epsilon_absolute + epsilon_relative * normOfAllLocalWeights

        println(f"Primal Residual: $primalResidual%.4f, Primal Tolerance: $primalTolerance%.4f")
        println(f"Dual Residual:   $dualResidual%.4f, Dual Tolerance:   $dualTolerance%.4f")

        primalResidual <= primalTolerance && dualResidual <= dualTolerance
    }

    /**
     * Automatically updates the penalty parameter `ρ` based on the ratio of primal and dual residuals.
     * Implements Equation (8) from paper section 3.4.
     */
    override def updateRho(
        currentRho: Double,
        primalResidual: Double,
        dualResidual: Double
    ): Double = {
        val mu = 10.0      // Balance parameter (recommended value)
        val tau_incr = 2.0 // Increment factor (recommended value)
        val tau_decr = 2.0 // Decrement factor (recommended value)

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
 * ============================================================================================
 * 3. ORCHESTRATION AND EXECUTION LOGIC
 * ============================================================================================
 */

/**
 * Orchestrates the distributed ADMM optimization process, as described in the paper.
 * This class corresponds to the `ADMMOptimizer` in the UML diagram. It manages the main
 * iteration loop, data persistence, and communication (broadcasts and aggregations).
 *
 * @param updater An instance of ADMMUpdater containing the problem-specific math.
 * @param numPartitions The number of partitions `N` to split the data into.
 * @param numFeatures The number of features `p` in the dataset.
 */
class ADMMOptimizer(

    val updater: ADMMUpdater,
    val numPartitions: Int,
    val numFeatures: Int) extends Serializable {

    def run(data: RDD[(Double, DenseVector[Double])], maxIterations: Int): DenseVector[Double] = {
        val sparkContext = data.sparkContext

        val partitionedData = data.repartition(numPartitions).persist(StorageLevel.MEMORY_AND_DISK)
        val numActualPartitions = partitionedData.getNumPartitions

        // Initialize ADMMState on each partition with zero vectors for w and u.
        var statesRDD = partitionedData.mapPartitions(_ => Iterator(ADMMState(
            localWeightVector_w = DenseVector.zeros[Double](numFeatures),
            scaledDualVariable_u = DenseVector.zeros[Double](numFeatures)
        )), preservesPartitioning = true).persist(StorageLevel.MEMORY_AND_DISK)

        // Initialize global variables on the driver
        var globalConsensusVariable_z = DenseVector.zeros[Double](numFeatures)
        var previousGlobalZ = DenseVector.zeros[Double](numFeatures)
        var penaltyParameter_rho = 1.0

        var isConverged = false
        var iteration = 0

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
            }.persist(StorageLevel.MEMORY_AND_DISK)

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
            }.persist(StorageLevel.MEMORY_AND_DISK)

            // Step 5: Check for convergence
            // Primal residual r^k = sqrt( Σ ||w_i - z||² )
            val primalResidual_squared_sum = states_after_u_update.map(s => norm(s.localWeightVector_w - globalConsensusVariable_z, 2)).map(n => n * n).sum()
            val primalResidual = math.sqrt(primalResidual_squared_sum)
            
            // Dual residual s^k = ρ * sqrt(N) * ||z^k - z^(k-1)||  (consensus form)
            val dualResidual = penaltyParameter_rho * math.sqrt(numActualPartitions) * norm(globalConsensusVariable_z - previousGlobalZ, 2)

            val normOfAllLocalWeights = math.sqrt(w_squared_norm_sum)
            val normOfGlobalZ = norm(globalConsensusVariable_z, 2)

            isConverged = updater.checkConvergence(primalResidual, dualResidual, normOfAllLocalWeights, normOfGlobalZ, numActualPartitions, numFeatures)

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
        globalConsensusVariable_z // The final model
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

    def train(data: RDD[(Double, DenseVector[Double])]): DenseVector[Double] = {
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
 * ============================================================================================
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
        }.persist(StorageLevel.MEMORY_AND_DISK)
        
        // Instantiate the user-facing algorithm class
        val admmLogisticRegression = new LogisticRegressionWithADMM(
            regularizationParam_lambda = lambda,
            numPartitions = numPartitions,
            maxIterations = maxIterations
        )

        // Train the model
        val finalModelWeights = admmLogisticRegression.train(data)

        println(s"\nFinal model weights (first 20):\n${finalModelWeights.toArray.take(20).mkString(",")}")

        // Save the final model weights to a file
        val writer = new PrintWriter(outputPath)
        try {
            writer.println(finalModelWeights.toArray.mkString(","))
        } finally {
            writer.close()
        }

        println(s"Model saved to $outputPath")
        spark.stop()
    }
}

/**
 * ============================================================================================
 * 5. CONVENIENCE FUNCTIONS FOR SPARK SHELL
 * ============================================================================================
 */

/**
 * Lists all available datasets in the sourced folder.
 */
def listDatasets(): Unit = {
  val sourcedPath = "/workspace/data/sourced"
  println(s"Available datasets in $sourcedPath:")
  println("=" * 50)
  
  try {
    val files = new java.io.File(sourcedPath).listFiles()
      .filter(_.isFile)
      .filter(f => f.getName.endsWith(".dat") || f.getName.endsWith(".csv"))
      .sortBy(_.getName)
    
    files.foreach { file =>
      val sizeInMB = file.length() / (1024 * 1024)
      println(f"  ${file.getName}%-35s (${sizeInMB}%3d MB)")
    }
    println()
    println("Usage: runADMM(\"filename.dat\") - will automatically look in sourced folder")
  } catch {
    case e: Exception => println(s"Error listing files: ${e.getMessage}")
  }
}

/**
 * Enhanced function to run ADMM on any dataset by filename.
 * Automatically looks in /workspace/data/sourced/ folder.
 */
def runADMM(filename: String, 
           numPartitions: Int = 4, 
           lambda: Double = 0.1, 
           maxIterations: Int = 50,
           outputPath: String = ""): DenseVector[Double] = {
  
  // Automatically prepend the sourced folder path if just filename is given
  val dataPath = if (filename.startsWith("/")) {
    filename // Full path provided
  } else {
    s"/workspace/data/sourced/$filename" // Just filename, add sourced path
  }
  
  println(s"Loading data from: $dataPath")
  val data = MLUtils.loadLibSVMFile(sc, dataPath).map {
    lp => (lp.label, DenseVector(lp.features.toArray))
  }.persist(StorageLevel.MEMORY_AND_DISK)
  
  // Automatically determine number of features from the data
  val n_features = data.map(_._2.length).max()
  println(s"Detected $n_features features in the dataset")
  
  // Count samples
  val numSamples = data.count()
  println(s"Dataset contains $numSamples samples")
  
  println(s"Running ADMM with:")
  println(s"  - Partitions: $numPartitions")  
  println(s"  - Lambda (regularization): $lambda")
  println(s"  - Max iterations: $maxIterations")
  println(s"  - Features: $n_features")
  
  // Use the proper LogisticRegressionWithADMM class
  val admmLogisticRegression = new LogisticRegressionWithADMM(
    regularizationParam_lambda = lambda,
    numPartitions = numPartitions,
    maxIterations = maxIterations
  )
  
  val startTime = System.currentTimeMillis()
  val final_model = admmLogisticRegression.train(data)
  val endTime = System.currentTimeMillis()
  
  val runTime = (endTime - startTime) / 1000.0
  println(s"\nADMM completed in $runTime seconds")
  println(s"Final model weights (first 10): ${final_model.toArray.take(10).mkString(", ")}")
  
  // Auto-generate output path if not provided
  val finalOutputPath = if (outputPath.isEmpty) {
    val baseFileName = filename.split("/").last.split("\\.").head
    s"/workspace/data/generated/${baseFileName}_admm_model.txt"
  } else {
    outputPath
  }
  
  // Save results
  val writer = new PrintWriter(finalOutputPath)
  writer.println(final_model.toArray.mkString(","))
  writer.close()
  println(s"Model saved to: $finalOutputPath")
  
  data.unpersist()
  final_model
}

/**
 * Interactive function that shows available datasets and prompts for selection.
 */
def runADMM_Interactive(): DenseVector[Double] = {
  listDatasets()
  print("Enter filename to run ADMM on: ")
  val filename = scala.io.StdIn.readLine()
  
  if (filename.trim.isEmpty) {
    println("No filename entered. Using sample_rcv1.dat as default.")
    runADMM("sample_rcv1.dat")
  } else {
    runADMM(filename.trim)
  }
}

/**
 * Quick test function with very few iterations for debugging.
 */
def testADMM(filename: String): DenseVector[Double] = {
  println(s"Running ADMM test on $filename (5 iterations only)...")
  runADMM(filename, numPartitions = 2, lambda = 0.1, maxIterations = 5)
}

/**
 * Function to run ADMM on the full RCV1 training dataset with optimized parameters.
 */
def runADMM_RCV1_Full(numPartitions: Int = 8): DenseVector[Double] = {
  println("Running ADMM on full RCV1 training dataset...")
  println("Note: This will take significantly longer than the sample dataset!")
  runADMM("lyrl2004_tokens_train.dat", numPartitions, lambda = 0.1, maxIterations = 50)
}

/**
 * Function to compare ADMM performance on different RCV1 dataset splits.
 */
def compareRCV1_Datasets(): Unit = {
  val rcv1Files = Array(
    "sample_rcv1.dat",
    "lyrl2004_tokens_train.dat",
    "lyrl2004_tokens_test_pt0.dat"
  )
  
  println("Comparing ADMM performance on different RCV1 datasets:")
  
  for (file <- rcv1Files) {
    println(s"\n=== Testing $file ===")
    try {
      val model = testADMM(file) // Use test version with fewer iterations
      println(s"Completed $file - Model norm: ${norm(model, 2)}")
    } catch {
      case e: Exception => println(s"Error with $file: ${e.getMessage}")
    }
  }
}

/**
 * Helper function to get dataset info without running ADMM.
 */
def datasetInfo(filename: String): Unit = {
  val dataPath = if (filename.startsWith("/")) filename else s"/workspace/data/sourced/$filename"
  
  try {
    println(s"Loading dataset info from: $dataPath")
    val data = MLUtils.loadLibSVMFile(sc, dataPath)
    
    val numSamples = data.count()
    val numFeatures = data.map(_.features.size).max()
    val labels = data.map(_.label).distinct().collect().sorted
    
    println(s"Dataset: $filename")
    println(s"  Samples: $numSamples")
    println(s"  Features: $numFeatures") 
    println(s"  Labels: ${labels.mkString(", ")}")
    
    data.unpersist()
  } catch {
    case e: Exception => println(s"Error reading $filename: ${e.getMessage}")
  }
}
