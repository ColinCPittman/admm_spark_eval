import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel

object Analysis {

  def runAnalysis(sc: SparkContext, fullDatasetPath: String, sampleFraction: Double = 0.2): Unit = {
    println("\n" + "="*80)
    println(s"Starting Scaled Replica Analysis (Sample: ${sampleFraction * 100}%)")
    println("="*80)

    // --- 1. Load and Sample the Data ---
    val fullData = MLUtils.loadLibSVMFile(sc, fullDatasetPath)
    val sampledData = fullData.sample(false, sampleFraction, seed = 42L).repartition(32).persist(StorageLevel.DISK_ONLY)
    
    val nSamples = sampledData.count()
    val nFeatures = sampledData.map(_.features.size).max()
    println(s"Using a ${sampleFraction * 100}% sample: $nSamples samples, $nFeatures features.")

    val splits = sampledData.randomSplit(Array(0.8, 0.2), seed = 11L)
    val trainingData = splits(0).persist(StorageLevel.DISK_ONLY)
    val testData = splits(1).persist(StorageLevel.DISK_ONLY)
    
    // --- 2. Run ADMM with Convergence Guard ---
    println("\n" + "-"*80)
    println("Running ADMM Implementation...")
    println("-" * 80)
    
    val admm = new LogisticRegressionWithADMM(
        regularizationParam_lambda = 0.1,
        numPartitions = 32,
        maxIterations = 50,
        minIterations = 10 // Force at least 10 iterations
    )

    val admmStartTime = System.currentTimeMillis()
    val admmModel = admm.train(trainingData.map(p => (p.label, breeze.linalg.DenseVector(p.features.toArray))))
    val admmRunTime = (System.currentTimeMillis() - admmStartTime) / 1000.0
    
    val admmAccuracy = calculateAccuracy(
        testData.map(p => (p.label, breeze.linalg.DenseVector(p.features.toArray))),
        admmModel
    ) * 100.0

    println(s"ADMM Completed in ${admmRunTime}s with Accuracy: ${admmAccuracy}%")


    // --- 3. Run MLLib Baseline for Comparison ---
    println("\n" + "-"*80)
    println("Running MLLib Baseline...")
    println("-" * 80)

    val mllibStartTime = System.currentTimeMillis()
    val mllibModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(trainingData)
    val mllibRunTime = (System.currentTimeMillis() - mllibStartTime) / 1000.0

    val predictionAndLabels = testData.map { case LabeledPoint(label, features) =>
      val prediction = mllibModel.predict(features)
      (prediction, label)
    }
    val mllibMetrics = new MulticlassMetrics(predictionAndLabels)
    val mllibAccuracy = mllibMetrics.accuracy * 100.0
    
    println(s"MLLib Completed in ${mllibRunTime}s with Accuracy: ${mllibAccuracy}%")

    // --- 4. Print Final Comparison Table ---
    println("\n" + "="*80)
    println("SCALED REPLICA RESULTS")
    println("="*80)
    println(f"Dataset Sample: ${sampleFraction * 100}% ($nSamples samples)")
    println(f"Partitions: 32")
    println("-" * 80)
    println(f"| Algorithm      | Runtime (s) | Accuracy (%%) |")
    println(f"|----------------|-------------|---------------|")
    println(f"| ADMM (Su's)    | ${admmRunTime}%-11.1f | ${admmAccuracy}%-13.2f |")
    println(f"| MLLib (LBFGS)  | ${mllibRunTime}%-11.1f | ${mllibAccuracy}%-13.2f |")
    println("="*80)

    // --- Cleanup ---
    sampledData.unpersist()
    trainingData.unpersist()
    testData.unpersist()
  }
}