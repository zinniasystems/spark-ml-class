package com.zinnia.ml.spark.ex2_logistic_regression

import org.apache.spark.SparkContext
import com.zinnia.ml.spark.util.RegressionUtil

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 22/1/14
 * Time: 12:27 PM
 * To change this template use File | Settings | File Templates.
 */
object Ex2_Data1{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("src/main/resources/ex2_logistic_regression/ex2data1.txt").cache()
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Running Regression")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMeanAndStdDev = featureScaledData._2
    val model = regression.runRegression(labelledRDD,10,Array(0.0,0.0),1.0,1.0)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    println("For a student with scores 85 and 45, we predict whether he would get admission or not as :")
    println(if(regression.doPrediction(model,Array(85.0,45.0),featureMeanAndStdDev) == 1.0) "Yes" else "No")
    println("For a student with scores 25 and 45, we predict whether he would get admission or not as :")
    println(if(regression.doPrediction(model,Array(25.0,45.0),featureMeanAndStdDev) == 1.0) "Yes" else "No")

  }
}

object Ex2_Data2{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex2_logistic_regression/ex2data2.txt").cache()
    println("Running Fourth Examples : ex2_logistic_regression Regression 2")
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Running Regression")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMeanAndStdDev = featureScaledData._2
    val model = regression.runRegression(labelledRDD,10,Array(0.0,0.0),20.0,1.0)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    println("Prediction for the values -0.21947,-0.016813 :")
    println(regression.doPrediction(model,Array(-0.21947,-0.016813),featureMeanAndStdDev))
    println("Prediction for the values 0.60426,0.59722 :")
    println(regression.doPrediction(model,Array(0.60426,0.59722),featureMeanAndStdDev))

  }
}
