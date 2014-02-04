package com.zinnia.ml.spark.ex7_pca

import org.apache.spark.SparkContext
import org.jblas.Singular
import com.zinnia.pca.pca

/**
 * Created with IntelliJ IDEA.
 * User: Shashi
 * Date: 4/2/14
 * Time: 4:43 PM
 * To change this template use File | Settings | File Templates.
 */
object ex7_pca_data1 {
    def main(args: Array[String]) {
      val sparkContext = new SparkContext("local", "run")
      val pca = new pca
      val data = sparkContext.textFile("src/main/resources/ex7_pca_data/ex7data1.txt")
      val inputRDD = pca.parseData(data)
      val meanStdTuple =  pca.computeMeanStd(inputRDD)
      pca.columnMean = meanStdTuple._1
      pca.columnStd = meanStdTuple._2
      val normalisedRDD = pca.getNormalisedRDD(inputRDD)
      val normalisedMeanStdTuple = pca.computeMeanStd(normalisedRDD)
      pca.normalisedColumnMean = normalisedMeanStdTuple._1
      pca.normalisedColumnStd = normalisedMeanStdTuple._2
      normalisedRDD.cache()
      val sigma = pca.getCovarianceMatrix(normalisedRDD)
      val svdOutput = Singular.fullSVD(sigma)
      val UMatrix = svdOutput(0)
      val SMatrix = svdOutput(1).toArray
      val K = pca.getKValue(SMatrix)
      pca.extractedUData = pca.extractUData(1,UMatrix)
      val projectedData = pca.projectData(normalisedRDD)
      var finalArray = List[String]()
      projectedData.foreach(array=>{
        var rowString = new String
        array.foreach(ele=>{
          rowString = rowString+","+ele.toString
        })
        finalArray = finalArray ::: List(rowString)
      })
      // Printing the projected data into text file
      sparkContext.makeRDD(finalArray).saveAsTextFile("src/main/resources/pca_output")
    }
}
