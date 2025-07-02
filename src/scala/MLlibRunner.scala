
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

object MLlibRunner {

  def runMLlib(sc: SparkContext, filename: String, numPartitions: Int): Unit = {
    println("\n" + "="*60)
    println(s"Running MLLib LogisticRegressionWithLBFGS on $filename")
    println("="*60)

    val dataPath = if (filename.startsWith("/")) filename else s"/workspace/data/sourced/$filename"
    val data = MLUtils.loadLibSVMFile(sc, dataPath).repartition(numPartitions)

    val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    val numSamples = data.count()
    val numFeatures = data.map(_.features.size).max()
    println(s"Dataset: $filename")
    println(s"  Samples: $numSamples, Features: $numFeatures, Partitions: $numPartitions")

    // Build the model
    val t0 = System.currentTimeMillis()
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(training)
    val t1 = System.currentTimeMillis()
    val runtimeSec = (t1 - t0) / 1000.0

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    
    println("\n" + ("-" * 60))
    println("MLLib Benchmark Results")
    println("-" * 60)
    printf("Runtime : %.1f s\n", runtimeSec)
    printf("Accuracy: %.2f %%\n", accuracy * 100.0)
    println(s"Partitions: $numPartitions")
    println("-" * 60)
  }
}
