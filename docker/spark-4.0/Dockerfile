FROM openjdk:17-jdk-slim

ENV SPARK_VERSION=4.0.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip:$PYTHONPATH

# Install required packages with retry logic for network issues
RUN apt-get update && \
    for i in {1..3}; do \
        apt-get install -y --no-install-recommends curl python3 iputils-ping procps && break || sleep 10; \
    done && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Download Spark with retry logic
RUN for i in {1..3}; do \
        curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && break || sleep 5; \
    done && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Download Breeze library for Scala 2.13 compatibility with Spark 4.0 (with retry)
RUN for i in {1..3}; do \
        curl -L -o $SPARK_HOME/jars/breeze_2.13-2.1.0.jar https://repo1.maven.org/maven2/org/scalanlp/breeze_2.13/2.1.0/breeze_2.13-2.1.0.jar && break || sleep 5; \
    done && \
    for i in {1..3}; do \
        curl -L -o $SPARK_HOME/jars/breeze-natives_2.13-2.1.0.jar https://repo1.maven.org/maven2/org/scalanlp/breeze-natives_2.13/2.1.0/breeze-natives_2.13-2.1.0.jar && break || sleep 5; \
    done

WORKDIR /workspace

CMD ["/bin/bash"] 