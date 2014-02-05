package com.zinnia.ml.spark.ex1_linear_regression

import org.apache.spark.SparkContext
import com.zinnia.ml.spark.util.RegressionUtil

/**
 * Created with IntelliJ IDEA.
 * User: Shashank L
 * Date: 22/1/14
 * Time: 12:26 PM
 * To change this template use File | Settings | File Templates.
 */
object Ex1 {
  def main(args: Array[String]) {
    val regression = new LinearRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("src/main/resources/ex1_linear_regression/ex1data1.txt").cache()
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Running Regression")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMean = featureScaledData._2
    val featureStd = featureScaledData._3
    val labelMean = featureScaledData._4
    val labelStd = featureScaledData._5
    val model = regression.runLinearRegression(labelledRDD, 30, Array(0.80),0.6,1.0)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    print("For population = 35,000, we predict a profit of: ")
    println(regression.doPrediction(model,Array(3.5),(labelMean,labelStd),featureMean,featureStd)*10000)
    println("Octave predicted value is: 2912.764904")
    print("For population = 75,000, we predict a profit of: ")
    println(regression.doPrediction(model,Array(7.0),(labelMean,labelStd),featureMean,featureStd)*10000)
    println("Octave predicted value is: 44606.906716")


  }
}

object Ex1_Multi {
  def main(args: Array[String]) {
    val regression = new LinearRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("src/main/resources/ex1_linear_regression/ex1data2.txt").cache()
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Running Regression")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMean = featureScaledData._2
    val featureStd = featureScaledData._3
    val labelMean = featureScaledData._4
    val labelStd = featureScaledData._5
    val model = regression.runLinearRegression(labelledRDD, 10, Array(0.0, 0.0),0.8,1.0)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    println("Predictions for the 1650 sqft 3 bed room house is: ")
    println(regression.doPrediction(model,Array(1650.0,3.0),(labelMean,labelStd),featureMean,featureStd))
    println("Octave result for this prediction was 293081.464335")

  }
}
