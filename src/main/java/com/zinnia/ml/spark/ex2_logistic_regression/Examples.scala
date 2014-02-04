package com.zinnia.ml.spark.ex2_logistic_regression

import org.apache.spark.SparkContext
import com.zinnia.ml.spark.util.RegressionUtil
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD

/**
 * Created with IntelliJ IDEA.
 * User: Shashank L
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
    println("Normalizing features")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMean = featureScaledData._2
    val featureStd = featureScaledData._3
    println("Running Regression")
    val model = regression.runRegression(labelledRDD,10,Array(0.0,0.0),0.5,1.0)
    println("Finding Error Rate")
    val errorRate = regression.computeCost(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    println("Predictions are")
    println("For a student with scores 85 and 45, we predict whether he would get admission or not as :")
    println(if(regression.doPrediction(model,Array(85.0,45.0),featureMean,featureStd) == 1.0) "Yes" else "No")
    println("For a student with scores 25 and 45, we predict whether he would get admission or not as :")
    println(if(regression.doPrediction(model,Array(25.0,45.0),featureMean,featureStd) == 1.0) "Yes" else "No")

  }
}


object Ex2_Data2_Poly{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("src/main/resources/ex2_logistic_regression/ex2data2.txt").cache()
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()

    println("Number of features before polynomial is: "+labelledRDD.first().features.length)
    val degreeOfPoly = 4
    labelledRDD = labelledRDD.map(eachPoint=>{
    def polynomialFeatures(featureArray:Array[Double],degree:Int):Array[Double] = {
      var outputArray = Array[Double]()
        for (i <- 1 to degree ;j<-0 to i){
          outputArray +:= scala.math.pow(featureArray(0),i-j)*scala.math.pow(featureArray(1),j)
        }
        outputArray
      }
      new LabeledPoint(eachPoint.label,polynomialFeatures(eachPoint.features,degreeOfPoly))
    })
    println("Number of features after polynomial is "+labelledRDD.first().features.length)
    println("Normalizing features")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMean = featureScaledData._2
    val featureStd = featureScaledData._3
    val thetaArray = new Array[Double](labelledRDD.first().features.length)
    println("Running Regression")
    //val model = regression.runRegression(labelledRDD,20,thetaArray,1.0,1.0)
    val model = regression.runRegularizedRegression(labelledRDD,20,thetaArray,1.0,1.0,0.05)
    println("Finding Error Rate")
    val errorRate = regression.computeCost(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    println("Prediction for the values -0.21947,-0.016813 :")
    println(regression.doPrediction(model,regression.polynomialFeatures(Array(-0.21947,-0.016813),degreeOfPoly),featureMean,featureStd))
    println("Prediction for the values 0.60426,0.59722 :")
    println(regression.doPrediction(model,regression.polynomialFeatures(Array(0.60426,0.59722),degreeOfPoly),featureMean,featureStd))

  }
}


