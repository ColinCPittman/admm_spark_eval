import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import breeze.linalg.DenseVector
import breeze.linalg.norm
import breeze.optimize.DiffFunction
import breeze.optimize.LBFGS
import breeze.numerics.sqrt
import org.apache.spark.storage.StorageLevel
import java.io.PrintWriter

/**
 * Holds the state for a single data partition in the ADMM algorithm.
 */
case class ADMMState(
    w: DenseVector[Double],
    u: DenseVector[Double]
)

/**
 * Contains the core mathematical logic for the ADMM updates.
 */
class ADMMUpdater(val lambda: Double) extends Serializable {

    /**
     * Performs the w-update on a single partition using L-BFGS.
     */
    def wUpdate(partitionData: Iterator[(Double, DenseVector[Double])],
                  currentState: ADMMState,
                  z_global: DenseVector[Double],
                  rho: Double): ADMMState = {

        val data = partitionData.toList
        if (data.isEmpty) return currentState

        val costFunction = new DiffFunction[DenseVector[Double]] {
            def calculate(w: DenseVector[Double]) = {
                val loss = data.map { case (y, x) =>
                    math.log(1 + math.exp(-y * (x.t * w)))
                }.sum / data.size

                val lagrangian = w - z_global + currentState.u
                val augmentedLagrangian = (rho / 2) * (lagrangian.t * lagrangian)

                val gradLoss = data.map { case (y, x) =>
                    x * (-y / (1 + math.exp(y * (x.t * w))))
                }.reduce(_ + _) / data.size.toDouble

                val gradLagrangian = lagrangian * rho

                (loss + augmentedLagrangian, gradLoss + gradLagrangian)
            }
        }

        val lbfgs = new LBFGS[DenseVector[Double]](maxIter = 100, m = 7)
        val new_w = lbfgs.minimize(costFunction, currentState.w)

        currentState.copy(w = new_w)
    }

    /**
     * Performs the z-update on the driver.
     */
    def zUpdate(w_avg: DenseVector[Double], u_avg: DenseVector[Double], rho: Double, numPartitions: Long): DenseVector[Double] = {
        val N = numPartitions.toDouble
        val factor = (N * rho) / ((N * rho) + (2 * lambda))
        (w_avg + u_avg) * factor
    }

    /**
     * Performs the u-update on a single partition.
     */
    def uUpdate(currentState: ADMMState, z_global: DenseVector[Double]): ADMMState = {
        val new_u = currentState.u + currentState.w - z_global
        currentState.copy(u = new_u)
    }

    /**
     * Checks for convergence based on primal and dual residuals.
     */
    def checkConvergence(primal_residual: Double, dual_residual: Double, w_norm: Double, z_norm: Double, numPartitions: Long, n_features: Int): Boolean = {
        val N = numPartitions.toDouble
        val n = n_features.toDouble
        val eps_abs = 1e-3
        val eps_rel = 1e-3

        val primal_tolerance = sqrt(N * n) * eps_abs + eps_rel * math.max(w_norm, z_norm)
        val dual_tolerance = sqrt(N * n) * eps_abs + eps_rel * w_norm

        println(f"Primal Residual: $primal_residual%.4f, Primal Tolerance: $primal_tolerance%.4f")
        println(f"Dual Residual:   $dual_residual%.4f, Dual Tolerance:   $dual_tolerance%.4f")

        primal_residual <= primal_tolerance && dual_residual <= dual_tolerance
    }
}

/**
 * Orchestrates the distributed ADMM optimization process.
 */
class ADMMOptimizer(val lambda: Double, val numPartitions: Int, val n_features: Int) {

    def run(data: RDD[(Double, DenseVector[Double])], maxIterations: Int): DenseVector[Double] = {
        val sc = data.sparkContext
        val updater = new ADMMUpdater(lambda)

        val partitionedData = data.repartition(numPartitions).persist(StorageLevel.MEMORY_AND_DISK)

        var states = partitionedData.mapPartitions(_ => Iterator(ADMMState(
            w = DenseVector.zeros[Double](n_features),
            u = DenseVector.zeros[Double](n_features)
        )), preservesPartitioning = true).persist(StorageLevel.MEMORY_AND_DISK)

        var z = DenseVector.zeros[Double](n_features)
        var z_prev = DenseVector.zeros[Double](n_features)
        var rho = 1.0

        var converged = false
        var iter = 0

        while (!converged && iter < maxIterations) {
            iter += 1
            println(s"\n--- Iteration: $iter ---")

            val z_b = sc.broadcast(z)
            val rho_b = sc.broadcast(rho)

            val dataAndStates = partitionedData.zipPartitions(states, preservesPartitioning = true) {
                (dataIter, stateIter) => Iterator((dataIter, stateIter.next))
            }

            val states_after_w = dataAndStates.map { case (partitionData, state) =>
                updater.wUpdate(partitionData, state, z_b.value, rho_b.value)
            }.persist(StorageLevel.MEMORY_AND_DISK)

            val (w_sum, u_sum) = states_after_w.map(s => (s.w, s.u)).treeAggregate(
                (DenseVector.zeros[Double](n_features), DenseVector.zeros[Double](n_features))
            )(
                seqOp = (agg, wu) => (agg._1 + wu._1, agg._2 + wu._2),
                combOp = (agg1, agg2) => (agg1._1 + agg2._1, agg1._2 + agg2._2)
            )
            val w_avg = w_sum / numPartitions.toDouble
            val u_avg = u_sum / numPartitions.toDouble

            z_prev = z
            z = updater.zUpdate(w_avg, u_avg, rho, numPartitions)
            val z_new_b = sc.broadcast(z)

            val states_after_u = states_after_w.map(state => updater.uUpdate(state, z_new_b.value)).persist(StorageLevel.MEMORY_AND_DISK)

            val primal_res_sq_sum = states_after_u.map(s => norm(s.w - z, 2)).map(n => n * n).sum()
            val primal_residual = math.sqrt(primal_res_sq_sum)
            val dual_residual = rho * norm(z - z_prev)
            val w_norm = norm(w_avg)
            val z_norm = norm(z)

            converged = updater.checkConvergence(primal_residual, dual_residual, w_norm, z_norm, numPartitions, n_features)

            val mu = 10.0
            val tau_incr = 2.0
            val tau_decr = 2.0

            if (primal_residual > mu * dual_residual) {
                rho = rho * tau_incr
            } else if (dual_residual > mu * primal_residual) {
                rho = rho / tau_decr
            }
            println(s"New Rho: $rho")

            states.unpersist()
            states_after_w.unpersist()
            states = states_after_u
            z_b.destroy()
            rho_b.destroy()
            z_new_b.destroy()
        }

        partitionedData.unpersist()
        states.unpersist()

        println(s"\nConvergence reached after $iter iterations.")
        z
    }
}

/**
 * A runnable object with a main method to load data, execute the optimizer, and save the results.
 */
object ADMMRunner {
    def main(args: Array[String]): Unit = {
        if (args.length < 4) {
            System.err.println("Usage: ADMMRunner.main(Array(<data_path>, <num_partitions>, <num_features>, <output_path>))")
            return
        }

        val dataPath = args(0)
        val numPartitions = args(1).toInt
        val n_features = args(2).toInt
        val outputPath = args(3)
        val lambda = 0.1
        val maxIterations = 50

        val sc = SparkSession.builder.getOrCreate().sparkContext

        println(s"Loading data from: $dataPath")
        val data = MLUtils.loadLibSVMFile(sc, dataPath).map {
            lp => (lp.label, DenseVector(lp.features.toArray))
        }

        println(s"Running ADMM with $numPartitions partitions, $n_features features.")
        val optimizer = new ADMMOptimizer(lambda, numPartitions, n_features)
        val final_model = optimizer.run(data, maxIterations)

        println(s"\nFinal model weights (first 20):\n${final_model.toArray.take(20).mkString(",")}")

        val writer = new PrintWriter(outputPath)
        writer.println(final_model.toArray.mkString(","))
        writer.close()

        println(s"Model saved to $outputPath")
    }
} 