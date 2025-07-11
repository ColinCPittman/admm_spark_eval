# An Empirical Evaluation of The Efficacy of an ADMM Classification Technique in Modern Spark

This repository contains the source code, data, and analysis for a CS 7265 Big Data Analytics group project. The timeline for this project is 6 weeks. 

**Based on the paper:** [Efficient Logistic Regression with L2 Regularization using ADMM on Spark (Su, 2020)](https://dl.acm.org/doi/10.1145/3409073.3409077)

**Team Members:**

* Colin Pittman
* Amonn Brewer
* Jennifer Felton
* Phillip Gregory 

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

1. **Get datasets** following instructions in `data/sourced/get-data/`

2. **Running the launcher:**
   
   ```bash
   launch-spark.bat
   ```

3. **Loading ADMM implementation in Spark shell:**

Spark 4.0.0:

```scala
:load /workspace/src/scala/SuADMM_40.scala
```

Spark 2.4.8:

```scala
:load /workspace/src/scala/SuADMM.scala
```

4. **Running experiments:**
   
   ```scala
   // ADMM test
   runADMM()
   ```

// MLlib LBFGS baseline test
runLBFGS()

```
**Dataset Selection:**
```scala
// RCV1
runADMM(dataset = "rcv1")
runLBFGS(dataset = "rcv1")

// HIGGS 
runADMM(dataset = "higgs") 
runLBFGS(dataset = "higgs")

// Custom file override, if needed.
runADMM(trainFilename = "custom_train.binary", testFilename = "custom_test.binary")
```

**Paper Reproduction Workflow:**

```scala
// Testing each partition counts used in Table 1 in Su's paper.
val partitions = List(5, 8, 10, 15)
for (p <- partitions) {
  println(s"\nTest with $p partitions:")
  runADMM(numPartitions = p)
  runLBFGS(numPartitions = p)
}
```

**Comparing Between Datasets:**

```scala
List("rcv1", "higgs").foreach { dataset =>
  println(s"\n=== Dataset: ${dataset.toUpperCase()} ===")
  runADMM(dataset = dataset)
  runLBFGS(dataset = dataset)
}
```

Both functions automatically save complete output information to `/data/generated/` with relevant filenames:

- ADMM: `{version}_{train}_{test}_admm_{run}_{partitions}part_output.txt`
- LBFGS: `{version}_{train}_{test}_lbfgs_{run}_{partitions}part_output.txt`

Example: `4_0_15k_10k_admm_1_8part_output.txt` (Spark 4.0, 15k train, 10k test, ADMM, run 1, 8 partitions)

Each output file contains:

- Dataset information and parameters
- Training progress and iteration details (for ADMM)
- Final results (accuracy, runtime, iterations)
- Model analysis (L2 norm, non-zero weights)
- Complete model weights

---

## Acknowledgments

Scala code implementations are adapted from [Spark-Optimization-ADMM](https://github.com/GMarzinotto/Spark-Optimization-ADMM) by GitHub user Gabriel Marzinotto.
