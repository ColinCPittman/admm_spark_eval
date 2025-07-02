# An Empirical Evaluation of The Efficacy of an ADMM Classification Technique in Modern Spark

This repository contains the source code, data, and analysis for a CS 7265 Big Data Analytics group project. The timeline for this project is 6 weeks. 

**Based on the paper:** [Efficient Logistic Regression with L2 Regularization using ADMM on Spark (Su, 2020)](https://dl.acm.org/doi/10.1145/3409073.3409077)

**Team Members:**

* Amonn Brewer
* Jennifer Felton
* Phillip Gregory
* Colin Pittman

---

## Project Goal

The purpose of this project is to evaluate the efficacy of the ADMM-based logistic regression algorithm proposed by Su (2020). Our research will benchmark the original implementation against modern Spark 4.0 MLlib, test its performance across different datasets, and explore potential algorithmic modifications.

The full plan is outlined in our [Project Proposal](/docs/project-proposal.pdf). Some scope adjustments may need to be made to this proposal to accommodate the project's accelerated timeline.

---

## Repository Structure

- `/data`: For datasets used in testing and generated/saved data for analysis.
- `/docker`: Self-contained Spark environments with auto-installers (Spark 2.4 and 4.0).
- `/docs`: For any project documentation, including the proposal and literature references.
- `/notebooks`: Jupyter/R notebooks for data analysis and result visualization.
- `/src`: All Scala source code for the Spark implementations.

---

## Getting Started

  The instructions in `data/sourced/get-data/` for acquiring the datasets.

  Run the launcher and choose your Spark version:

```bash
launch-spark.bat
```

  The launcher *should* automatically:

- Install Docker Desktop if needed
- Start Docker in the background  
- Build Spark images (first run: ~5-10 minutes)
- Launch the appropriate environment

Datasets are available at `/workspace/data/sourced/` inside the container.

To load the ADMM implementation, use `:load` instead of `:paste` to avoid serialization issues:

- **Spark 4.0**: `:load /workspace/src/scala/ADMM.scala`
- **Spark 2.4**: `:load /workspace/src/scala/SuADMM.scala`

---

## Running SuADMM.scala in Spark 2.4 shell

After loading `SuADMM.scala` in the Spark 2.4 shell, several convenience functions are available:

### Basic Usage

```scala
// List all available datasets with file sizes
listDatasets()

// Run ADMM on any dataset (looks in /workspace/data/sourced/ automatically)
runADMM("sample_rcv1.dat")

// Run ADMM with custom parameters  
runADMM("sample_rcv1.dat", numPartitions = 8, lambda = 0.05, maxIterations = 100)
```

### Interactive Mode

```scala
// Prompts you to select from available datasets
runADMM_Interactive()
```

### Testing & Debugging

```scala
// Quick test with only 5 iterations (faster for debugging)
testADMM("sample_rcv1.dat")

// Get dataset info without running ADMM
datasetInfo("sample_rcv1.dat")
```

### Specialized Functions

```scala
// Run on full RCV1 training dataset with optimized parameters
runADMM_RCV1_Full()

// Compare ADMM performance across multiple RCV1 datasets
compareRCV1_Datasets()
```

### Paper Reproduction & Accuracy Evaluation

```scala
// Enhanced ADMM with accuracy calculation (reproduces paper's experimental setup)
runADMM_WithAccuracy("lyrl2004_vectors_test_pt0.dat")

// Direct comparison between ADMM and MLlib (reproduces Table 1 from paper)
compareADMM_vs_MLlib("lyrl2004_vectors_test_pt0.dat", numPartitions = 8)

// Reproduce the exact experimental setup from Su et al. Table 1
reproduceTable1()

// Calculate accuracy for any trained model
calculateAccuracy(testData, trainedModel)

// Split dataset into training/testing sets (80/20 split like the paper)
val (trainData, testData) = splitTrainTest(fullData, trainRatio = 0.8)
```


### Parameters

- `filename`: Dataset filename (automatically looks in `/workspace/data/sourced/`)
- `numPartitions`: Number of Spark partitions (default: 4)
- `lambda`: L2 regularization parameter (default: 0.1, as per Su et al. paper)
- `maxIterations`: Maximum ADMM iterations (default: 50)
- `outputPath`: Where to save model weights (auto-generated if not specified)

Models are automatically saved to `/workspace/data/generated/` with descriptive filenames.


## Acknowledgments

Scala code implementations are adapted from [Spark-Optimization-ADMM](https://github.com/GMarzinotto/Spark-Optimization-ADMM) by GitHub user Gabriel Marzinotto.
