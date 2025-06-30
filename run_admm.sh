#!/bin/bash

# Compile the Scala code
echo "Compiling ADMM.scala..."
cd /workspace
scalac -cp "$(find /opt/spark/jars -name '*.jar' | tr '\n' ':')." src/scala/ADMM.scala -d /tmp/classes

# Create JAR
echo "Creating JAR..."
mkdir -p /tmp/classes
cd /tmp/classes
jar cf /tmp/admm.jar .

# Run with spark-submit
echo "Running ADMM with spark-submit..."
cd /workspace
spark-submit \
  --class ADMMRunner \
  --master local[*] \
  /tmp/admm.jar \
  /workspace/data/sourced/sample_higgs.csv \
  4 \
  28 \
  /workspace/data/generated/admm_model.txt 