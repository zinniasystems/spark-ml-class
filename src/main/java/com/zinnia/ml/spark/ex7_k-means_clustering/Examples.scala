package com.zinnia.ml.spark.ex7_k

import org.apache.spark.SparkContext
import com.zinnia.ml.spark.util.RegressionUtil
import org.apache.spark.mllib.clustering


/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 31/1/14
 * Time: 4:25 PM
 * To change this template use File | Settings | File Templates.
 */
object Ex7Data1 {
  def main(args: Array[String]) {
    val regression = new KMeans
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex7_K-Means/ex7data1.csv").cache()
    val inputDataRDD = fileContents.map(eachString=>{
      eachString.split(',').map(x => x.toDouble).toArray
    })
    val kMeans = new clustering.KMeans()
    kMeans.setK(3)
    kMeans.setMaxIterations(100)
    val model = kMeans.run(inputDataRDD)
    println(model.computeCost(inputDataRDD))
    model.clusterCenters.foreach(each=>{
      println("Cluster ")
      each.foreach(println)
    })
  }
}


object Ex7ImageData{
  def main (args: Array[String]) {
    val regression = new KMeans
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex7_K-Means/imageData.csv").cache()
    val inputDataRDD = fileContents.map(eachString=>{
      eachString.split(',').map(x => x.toDouble).toArray
    })

    val kMeans = new clustering.KMeans()
    kMeans.setK(16)
    kMeans.setMaxIterations(10)
    val model = kMeans.run(inputDataRDD)
    println(model.computeCost(inputDataRDD))
    model.clusterCenters.foreach(each=>{
      println("Cluster ")
      each.foreach(println)
    })
  }
}
