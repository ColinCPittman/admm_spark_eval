FROM openjdk:8-jdk-slim

ENV SPARK_VERSION=2.4.8
ENV HADOOP_VERSION=2.7
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.7-src.zip:$PYTHONPATH

RUN apt-get update && \
    apt-get install -y curl iputils-ping procps netcat-openbsd && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Remove old/conflicting Breeze versions that ship with Spark 2.4.8
RUN rm -f $SPARK_HOME/jars/breeze*

# Add essential Breeze JAR for Scala 2.11 (required for ADMM implementation)
RUN curl -L -o $SPARK_HOME/jars/breeze_2.11-1.0.jar \
    https://repo1.maven.org/maven2/org/scalanlp/breeze_2.11/1.0/breeze_2.11-1.0.jar

WORKDIR /workspace

CMD ["/bin/bash"] 