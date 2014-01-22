package com.zinnia.ml.spark.ex1_linear_regression

import org.apache.spark.SparkContext
import com.zinnia.ml.spark.util.RegressionUtil

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 22/1/14
 * Time: 12:26 PM
 * To change this template use File | Settings | File Templates.
 */
object Ex1 {
  def main(args: Array[String]) {
    val regression = new LinearRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex1_linear_regression/ex1data1.txt").cache()
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Running Regression")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val labelMeanAndStdDev = featureScaledData._2
    val featureMeanAndStdDev = featureScaledData._3
    labelledRDD = featureScaledData._1
    val model = regression.runLinearRegression(labelledRDD, 30, Array(0.80),0.6,1.0)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    print("For population = 35,000, we predict a profit of: ")
    println(regression.doPrediction(model,Array(3.5),labelMeanAndStdDev,featureMeanAndStdDev)*10000)
    println("Octave predicted value is: 2912.764904")
    print("For population = 75,000, we predict a profit of: ")
    println(regression.doPrediction(model,Array(7.0),labelMeanAndStdDev,featureMeanAndStdDev)*10000)
    println("Octave predicted value is: 44606.906716")


  }
}

object Ex1_Multi {
  def main(args: Array[String]) {
    val regression = new LinearRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex1_linear_regression/ex1data2.txt").cache()
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Running Regression")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val labelMeanAndStdDev = featureScaledData._2
    val featureMeanAndStdDev = featureScaledData._3
    val model = regression.runLinearRegression(labelledRDD, 10, Array(0.0, 0.0),0.8,1.0)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    println("Predictions for the 1650 sqft 3 bed room house is: ")
    println(regression.doPrediction(model,Array(1650.0,3.0),labelMeanAndStdDev,featureMeanAndStdDev))
    println("Octave result for this prediction was 293081.464335")

  }
}
