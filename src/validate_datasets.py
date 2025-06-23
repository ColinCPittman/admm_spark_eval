#!/usr/bin/env python3
"""
Validate dataset access and basic properties for ADMM experiments.
Run this to verify the Docker environment can handle the datasets.
"""

from pyspark.sql import SparkSession
import os
import time

def main():
    # Initialize Spark with appropriate memory settings
    spark = SparkSession.builder \
        .appName("Dataset Validation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .getOrCreate()
    
    print(f"Spark Version: {spark.version}")
    print(f"Available cores: {spark.sparkContext.defaultParallelism}")
    print("-" * 50)
    
    # Test HIGGS dataset
    higgs_path = "/workspace/data/sourced/HIGGS.csv"
    if os.path.exists(higgs_path):
        print(f"Loading HIGGS dataset ({os.path.getsize(higgs_path) / (1024**3):.1f} GB)...")
        start_time = time.time()
        
        higgs_df = spark.read.csv(higgs_path, header=False, inferSchema=True)
        higgs_df.cache()  # Cache for performance
        
        count = higgs_df.count()
        cols = len(higgs_df.columns)
        load_time = time.time() - start_time
        
        print(f"✅ HIGGS: {count:,} rows, {cols} columns")
        print(f"   Load time: {load_time:.1f}s")
        
        # Show sample
        print("   Sample data:")
        higgs_df.show(3, truncate=False)
    else:
        print("❌ HIGGS.csv not found")
    
    print("-" * 50)
    
    # Test RCV1 datasets
    rcv1_files = [
        "lyrl2004_vectors_test_pt0.dat",
        "lyrl2004_vectors_test_pt1.dat", 
        "lyrl2004_tokens_train.dat"
    ]
    
    for file in rcv1_files:
        file_path = f"/workspace/data/sourced/{file}"
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024**2)
            print(f"✅ {file}: {size_mb:.1f} MB")
        else:
            print(f"❌ {file}: Not found")
    
    spark.stop()
    print("\nValidation complete!")

if __name__ == "__main__":
    main() 