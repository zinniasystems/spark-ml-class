package com.zinnia.ml.spark.ex1_linear_regression

import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint, LinearRegressionModel}
import org.apache.spark.rdd.RDD
import java.lang.Math
import com.zinnia.ml.spark.util.RegressionUtil
import org.apache.spark.mllib.util.MLUtils

/**
 * Created with IntelliJ IDEA.
 * User: Shashank L
 * Date: 21/1/14
 * Time: 11:27 PM
 * To change this template use File | Settings | File Templates.
 */
class LinearRegression {

  //Predicts the possibility of the data set provided as input and handles feature normalisations also.
  def doPrediction(model:LinearRegressionModel, features:Array[Double],labelMeanStd:(Double,Double)=null,featureMean:Array[Double]=null, featureStdDev : Array[Double]=null):Double={
    if (labelMeanStd !=null || featureMean != null || featureStdDev != null) {
      val normalizedFeatures = new Array[Double](features.length)
      for (a <- 0 to features.length - 1) {
        normalizedFeatures(a) = (features(a) - featureMean(a)) / featureStdDev(a)
      }
      val finalPredictedValue = model.predict(normalizedFeatures) * labelMeanStd._2 + labelMeanStd._1
      finalPredictedValue
    } else {
      val finalPredictedValue = model.predict(features)
      finalPredictedValue
    }

  }

  //Calculates the rate of error the model is predicting
  def findErrorRate(labelledRDD: RDD[LabeledPoint], model: LinearRegressionModel): Double = {
    val total = labelledRDD.map (
      labeledPoint =>{
        val prediction = model.predict(labeledPoint.features)
        Math.pow(labeledPoint.label - prediction, 2.0)
    }).reduce(_+_)
    val trainError = total / (2 * labelledRDD.count())
    trainError
  }


  //Runs ex1_linear_regression Regression on the RDD with the optimizations passed as parameter and returns a LR model which can be used for predictions
  def runLinearRegression(labelledRDD: RDD[LabeledPoint], numberOfIterations: Int, initialWeights:Array[Double], stepSize:Double,miniBatchFraction:Double): LinearRegressionModel = {
    val regression = new LinearRegressionWithSGD()
    regression.optimizer.setNumIterations(numberOfIterations).setStepSize(stepSize).setMiniBatchFraction(miniBatchFraction)
    val linearRegressionModel = regression.run(labelledRDD, initialWeights)
    linearRegressionModel
  }

  //Normalises any number of features in a RDD of LabeledPoints and returns the normalised RDD, mean and std dev of label and features.
  def normaliseFeatures(labelledRDD: RDD[LabeledPoint]): (RDD[LabeledPoint], Array[Double], Array[Double], Double, Double) = {
    val context = labelledRDD.context
    val numOfFeatures = labelledRDD.first().features.length
    val nexamples = labelledRDD.count()
    val results = MLUtils.computeStats(labelledRDD,numOfFeatures,nexamples)
    val labelMean = results._1
    val featureMean = results._2.toArray
    val featureStdDev = results._3.toArray

    var broadcastLabelMean = context.broadcast(labelMean)

    val intermediate = labelledRDD.map(eachLabelledPoint=>{
      Math.pow(eachLabelledPoint.label-broadcastLabelMean.value,2.0)
    }).reduce((a,b)=>{a+b})

    val labelStdDev = Math.sqrt(intermediate/(nexamples-1))
    val broadcastMeanAndStdDev = context.broadcast(labelMean,labelStdDev,featureMean,featureStdDev)

    val normalizedRDD = labelledRDD.map(point => {
      val normalizedFeatureArray = new Array[Double](numOfFeatures)
      val features = point.features
      for (a <- 0 to numOfFeatures - 1) {
        if (broadcastMeanAndStdDev.value._3(a) == 0 && broadcastMeanAndStdDev.value._4(a) == 0) {
          normalizedFeatureArray(a) = 0.0
        } else {
          normalizedFeatureArray(a) = (features(a) - broadcastMeanAndStdDev.value._3(a)) / broadcastMeanAndStdDev.value._4(a)
        }

      }
      LabeledPoint((point.label - broadcastMeanAndStdDev.value._1)/broadcastMeanAndStdDev.value._2, normalizedFeatureArray)
    })
    (normalizedRDD,featureMean,featureStdDev,labelMean,labelStdDev)
  }


}
