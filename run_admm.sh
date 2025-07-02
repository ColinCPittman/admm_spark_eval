#!/bin/bash

# Compile the Scala code
echo "Compiling SuADMM.scala..."
cd /workspace
scalac -cp "$(find /opt/spark/jars -name '*.jar' | tr '\n' ':')." src/scala/SuADMM.scala -d /tmp/classes

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
  /workspace/data/sourced/sample_rcv1.dat \
  8 \
  47236 \
  /workspace/data/generated/sample_rcv1_admm_model.txt 