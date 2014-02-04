package com.zinnia.ml.spark.ex7.K_Means_Clustering

import org.apache.spark.SparkContext
import com.zinnia.spark.machinelearning.CustomLinearRegression
import org.apache.spark.mllib.clustering.KMeans

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 23/1/14
 * Time: 1:22 PM
 * To change this template use File | Settings | File Templates.
 */
object Main {
  def main(args:Array[String]){
    val sparkContext = new SparkContext("local", "kmeans")
    val customKmeans = new CustomKmeans()

    val inputFileex7 = "src/main/resources/ex7data1.txt"
    val inputRawRDDex7 = sparkContext.textFile(inputFileex7)
    val inputRDDex7=customKmeans.getInputDataSet(inputRawRDDex7)
    val noOfCluster1 = 3
    customKmeans.runKmeans(inputRDDex7,noOfCluster1)

    val inputFile = "src/main/resources/bird_data.txt"
    val inputRawRDD = sparkContext.textFile(inputFile)
    val inputRDD=customKmeans.getInputDataSet(inputRawRDD)
    val noOfCluster = 16
    val centroids = customKmeans.getCentroids(inputRDD,noOfCluster)
    val retrievedPoints =  customKmeans.mapCentroidForEachCluster(centroids,inputRDD)
    retrievedPoints.saveAsTextFile("/home/hadoop/desktop/compressed.txt")
  }
}
