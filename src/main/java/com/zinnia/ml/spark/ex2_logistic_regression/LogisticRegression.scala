package com.zinnia.ml.spark.ex2_logistic_regression

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import com.zinnia.ml.spark.util.RegressionUtil

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 21/1/14
 * Time: 11:26 PM
 * To change this template use File | Settings | File Templates.
 */
class LogisticRegression {

  def doPrediction(model:LogisticRegressionModel, features:Array[Double],featureMeanStd:Array[(Double,Double)] = null):Double={
    if (featureMeanStd != null) {
      val normalizedFeatures = new Array[Double](features.length)
      for (a <- 0 to features.length - 1) {
        normalizedFeatures(a) = (features(a) - featureMeanStd(a)._1) / featureMeanStd(a)._2
      }
      val finalPredictedValue = model.predict(normalizedFeatures)
      finalPredictedValue
    } else {
      val finalPredictedValue = model.predict(features)
      finalPredictedValue
    }

  }

  def findErrorRate(labelledRDD: RDD[LabeledPoint], model: LogisticRegressionModel): Double = {
    val labelAndPreds = labelledRDD.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / labelAndPreds.count
    trainErr
  }

  def runRegression(labelledRDD: RDD[LabeledPoint], numberOfIterations: Int, initialWeights:Array[Double], stepSize:Double,miniBatchFraction:Double): LogisticRegressionModel = {
    val regression = new LogisticRegressionWithSGD()
    regression.optimizer.setNumIterations(numberOfIterations).setStepSize(stepSize).setMiniBatchFraction(miniBatchFraction)
    val logisticRegressionModel = regression.run(labelledRDD,initialWeights)
    logisticRegressionModel
  }

  def normaliseFeatures(labelledRDD: RDD[LabeledPoint]): (RDD[LabeledPoint],Array[(Double,Double)]) = {
    val numOfFeatures = labelledRDD.first().features.length
    val featureMeanAndStdDev = new Array[(Double,Double)](numOfFeatures)

    for (a <- 0 to labelledRDD.first().features.length-1){
      val singleFeature = labelledRDD.map(point => {
        point.features(a)
      })
      singleFeature.cache()
      featureMeanAndStdDev(a) = (RegressionUtil.calcMeanAndStdDev(singleFeature.toArray()))
    }

    val normalizedRDD = labelledRDD.map(point => {
      var normalizedFeatureArray = new Array[Double](numOfFeatures)
      val features = point.features
      for (a <- 0 to numOfFeatures-1) {
        normalizedFeatureArray(a) = (features(a) - featureMeanAndStdDev(a)._1) / featureMeanAndStdDev(a)._2
      }
      LabeledPoint(point.label, normalizedFeatureArray)
    })
    (normalizedRDD,featureMeanAndStdDev)
  }

}
