// MLlibRunner.scala

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import java.io.PrintWriter

object MLlibRunner {
  def main(args: Array[String]): Unit = {
    if (args.length < 2) {
      System.err.println("Usage: MLlibRunner <data_path> <output_path>")
      System.exit(1)
    }

    val dataPath = args(0)
    val outputPath = args(1)

    // 1. Initialize Spark Session
    val spark = SparkSession.builder
      .appName("MLlib Logistic Regression Baseline")
      .getOrCreate()
    val sc = spark.sparkContext

    println(s"Loading data from: $dataPath")

    // 2. Load the data in LIBSVM format
    // This is the standard format MLlib works with.
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, dataPath)
    
    // MLlib's logistic regression handles {-1, +1} labels, but often 
    // works best with {0, 1}. We'll ensure labels are {0,1} for stability,
    // as MLlib's binary classifier expects this.
    val mappedData = data.map(lp => LabeledPoint(if (lp.label > 0) 1.0 else 0.0, lp.features))
    
    // The data is cached for iterative access by the L-BFGS algorithm
    mappedData.cache()

    // 3. Configure and run the MLlib Logistic Regression
    println("Training model with mllib.LogisticRegressionWithLBFGS...")
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(mappedData)
    
    println("Model training complete.")

    // 4. Extract and save the final model weights
    val final_weights = model.weights.toArray

    println(s"\nFinal model weights (first 20):\n${final_weights.take(20).mkString(",")}")

    val writer = new PrintWriter(outputPath)
    writer.println(final_weights.mkString(","))
    writer.close()

    println(s"MLlib model saved to $outputPath")

    spark.stop()
  }
}
