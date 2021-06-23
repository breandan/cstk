package edu.mcgill.gymfs.experiments

//import org.tribuo.*
//import org.tribuo.clustering.ClusteringFactory
//import org.tribuo.clustering.example.ClusteringDataGenerator
import org.tribuo.*
import org.tribuo.clustering.example.ClusteringDataGenerator
import org.tribuo.clustering.kmeans.KMeansTrainer
import org.tribuo.clustering.kmeans.KMeansTrainer.Distance.EUCLIDEAN
import kotlin.random.Random
import kotlin.system.measureTimeMillis

fun main() {
  println("" + measureTimeMillis {
    val data = ClusteringDataGenerator.gaussianClusters(500, 1L)
    val test = ClusteringDataGenerator.gaussianClusters(500, 2L)
    val trainer = KMeansTrainer(5, 10, EUCLIDEAN, 1, 1)
    val model = trainer.train(data)
    model.predict(test).forEach { println(it.output.id) }
  } + "ms")
}