package com.zinnia.ml.spark.ex1_linear_regression

import org.apache.spark.mllib.regression.{LinearRegressionWithSGD, LabeledPoint, LinearRegressionModel}
import org.apache.spark.rdd.RDD
import java.lang.Math
import com.zinnia.ml.spark.util.RegressionUtil

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 21/1/14
 * Time: 11:27 PM
 * To change this template use File | Settings | File Templates.
 */
class LinearRegression {

  //Predicts the possibility of the data set provided as input and handles feature normalisations also.
  def doPrediction(model:LinearRegressionModel, features:Array[Double],labelMeanStd:(Double,Double)=null,featureMeanStd:Array[(Double,Double)]=null):Double={
    if (labelMeanStd !=null && featureMeanStd != null) {
      val normalizedFeatures = new Array[Double](features.length)
      for (a <- 0 to features.length - 1) {
        normalizedFeatures(a) = (features(a) - featureMeanStd(a)._1) / featureMeanStd(a)._2
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
    val labelAndPreds = labelledRDD.map {
      labeledPoint =>
        val prediction = model.predict(labeledPoint.features)
        Math.pow(labeledPoint.label - prediction, 2.0)
    }
    val total = labelAndPreds.reduce((firstValue, secondValue) => {
      firstValue + secondValue
    })
    val trainError = total / (2 * labelAndPreds.count())
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
  def normaliseFeatures(labelledRDD: RDD[LabeledPoint]): (RDD[LabeledPoint],(Double,Double),Array[(Double,Double)]) = {
    val numOfFeatures = labelledRDD.first().features.length
    val featureMeanAndStdDev = new Array[(Double,Double)](numOfFeatures)

    for (a <- 0 to labelledRDD.first().features.length-1){
      val singleFeature = labelledRDD.map(point => {
        point.features(a)
      })
      singleFeature.cache()
      featureMeanAndStdDev(a) = (RegressionUtil.calcMeanAndStdDev(singleFeature.toArray()))
    }

    val label = labelledRDD.map(point => {
      point.label
    })

    var labelMeanAndStdDev = RegressionUtil.calcMeanAndStdDev(label.toArray())

    val normalizedRDD = labelledRDD.map(point => {
      var normalizedFeatureArray = new Array[Double](numOfFeatures)
      val features = point.features
      for (a <- 0 to numOfFeatures-1) {
        normalizedFeatureArray(a) = (features(a) - featureMeanAndStdDev(a)._1) / featureMeanAndStdDev(a)._2
      }
      val newLabel = (point.label - labelMeanAndStdDev._1) / labelMeanAndStdDev._2
      LabeledPoint(newLabel, normalizedFeatureArray)
    })
    (normalizedRDD,labelMeanAndStdDev,featureMeanAndStdDev)
  }

  /*def scaleSingleFeature(labelledRDD: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val feature = labelledRDD.map(point => {
      point.features(0)
    })
    val label = labelledRDD.map(point => {
      point.label
    })
    val meanAndStdLabel = calcMeanAndStdDev(label.toArray())
    val meanAndStd = calcMeanAndStdDev(feature.toArray())
    val featurizedRDD = labelledRDD.map(value => {
      LabeledPoint(((value.label - meanAndStdLabel._2) / meanAndStdLabel._1), Array(((value.features(0) - meanAndStd._2) / meanAndStd._1)))
    })
    featurizedRDD
  }*/

}
