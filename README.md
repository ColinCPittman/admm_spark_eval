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

Currently :load is facing issues with scope, but `:paste /workspace/src/scala/ADMM.scala` does work.

## Acknowledgments

Scala code implementations are adapted from [Spark-Optimization-ADMM](https://github.com/GMarzinotto/Spark-Optimization-ADMM) by GitHub user Gabriel Marzinotto.
