name := "admm-spark-eval"

version := "1.0"

// Use environment variable to determine versions, defaulting to Spark 4.0
lazy val sparkVersion = sys.env.getOrElse("SPARK_VERSION", "4.0.0")
lazy val isSparkOld = sparkVersion.startsWith("2.")

// Scala version compatibility
scalaVersion := (if (isSparkOld) "2.12.15" else "2.12.17")

// Dynamic dependency versions based on Spark version
lazy val breezeVersion = if (isSparkOld) "1.0" else "2.1.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.scalanlp" %% "breeze" % breezeVersion
)

// Java compatibility settings
javacOptions ++= {
  if (isSparkOld) Seq("-source", "8", "-target", "8")
  else Seq("-source", "17", "-target", "17")
}

// Scala compiler options
scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked"
)

// Assembly settings for creating fat JARs
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

// Print current configuration
initialize := {
  val _ = initialize.value
  println(s"Building with Spark $sparkVersion and Scala ${scalaVersion.value}")
  println(s"Breeze version: $breezeVersion")
} 