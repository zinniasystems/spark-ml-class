package com.zinnia.ml.spark.util

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.util.grammar.LabelledRHS
import org.apache.spark.mllib.regression.{LinearRegressionModel, LinearRegressionWithSGD, LabeledPoint}
import java.lang.Math
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionModel}

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 20/1/14
 * Time: 4:13 PM
 * To change this template use File | Settings | File Templates.
 */

object RegressionUtil {

  //Function to calculate the standard deviation and mean of the given array of values.
  def calcMeanAndStdDev(numbers: Array[Double]): (Double, Double) = {
    var sum: Double = 0.0;
    var mean = 0.0
    if (numbers.length >= 2) {
      mean = numbers.reduceLeft(_ + _) / numbers.length
      val factor: Double = 1.0 / (numbers.length.toDouble - 1)
      for (x: Double <- numbers) {
        sum = sum + ((x - mean) * (x - mean))
      }
      sum = sum * factor
    }
    (mean,Math.sqrt(sum))
  }

  //parses a csv file where each line has a set of features at the beginning and the label at the end
  def parseFileContent(inputData: RDD[String]): RDD[LabeledPoint] = {

    val labelledRDD = inputData.map(line => {
      val parts = line.split(",")
      LabeledPoint(parts.last.toDouble, parts.init.map(x => x.toDouble).toArray)
    })
    labelledRDD

  }

  def splitData(inputData: RDD[String], percentageOfData:Double) : RDD[String] = {
    val numberOfValues = inputData.count() * percentageOfData

    var i = 0.0
    inputData.filter(each=>{
      i = i+1.0
      if(i < numberOfValues)
        true
      else
        false

    })
  }

}



