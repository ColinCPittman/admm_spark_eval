FROM openjdk:17-jdk-slim

ENV SPARK_VERSION=4.0.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip:$PYTHONPATH

RUN apt-get update && \
    apt-get install -y curl python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} $SPARK_HOME && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# Download Breeze library for Scala 2.13 compatibility with Spark 4.0
RUN curl -L -o $SPARK_HOME/jars/breeze_2.13-2.1.0.jar https://repo1.maven.org/maven2/org/scalanlp/breeze_2.13/2.1.0/breeze_2.13-2.1.0.jar && \
    curl -L -o $SPARK_HOME/jars/breeze-natives_2.13-2.1.0.jar https://repo1.maven.org/maven2/org/scalanlp/breeze-natives_2.13/2.1.0/breeze-natives_2.13-2.1.0.jar

RUN pip3 install pyspark==${SPARK_VERSION} numpy pandas

WORKDIR /workspace

CMD ["/bin/bash"] 