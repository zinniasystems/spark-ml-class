package com.zinnia.ml.spark.ex2_logistic_regression

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionModel}
import org.apache.spark.rdd.RDD
import org.jblas.DoubleMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import java.lang.Math
import org.apache.spark.mllib.optimization.L1Updater
import java.util.Random
import org.apache.spark.mllib.util.MLUtils

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 21/1/14
 * Time: 11:26 PM
 * To change this template use File | Settings | File Templates.
 */
class LogisticRegression {

  def runMultiClassRegression(listOfMappedRDDs: List[(Int, RDD[LabeledPoint])], numberOfIterations: Int, initialWeights: Array[Double], stepSize: Double, miniBatchFraction: Double): Map[Int, LogisticRegressionModel] = {
    val broadcastRegressionOptions = listOfMappedRDDs.head._2.context.broadcast(numberOfIterations,initialWeights,stepSize,miniBatchFraction)
    val rddOfModels = listOfMappedRDDs.map(eachRDD => {
      val regression = new LogisticRegressionWithSGD()
      regression.optimizer.setNumIterations(broadcastRegressionOptions.value._1).setStepSize(broadcastRegressionOptions.value._3).setMiniBatchFraction(broadcastRegressionOptions.value._4)
      val model = regression.run(eachRDD._2, broadcastRegressionOptions.value._2)
      (eachRDD._1, model)
    })
    rddOfModels.toMap
  }

  def findErrorRateMultiClass(modelMap: Map[Int, LogisticRegressionModel], labelledRDD: RDD[LabeledPoint]): Double = {
    val broadcastModelMap = labelledRDD.context.broadcast(modelMap)
    val labelAndPreds = labelledRDD.map(point => {
      def predictPoint(model: LogisticRegressionModel, testData: Array[Double]): Double = {
        val testDataMat = new DoubleMatrix(1, testData.length, testData: _*)
        val weightMatrix = new DoubleMatrix(model.weights.length, 1, model.weights: _*)
        val margin = testDataMat.mmul(weightMatrix).get(0) + model.intercept
        val value = 1.0 / (1.0 + math.exp(margin * -1))
        value
      }
      var resultsMap: Map[Int, Double] = Map()
      val modelMap = broadcastModelMap.value
      for (i <- 1 to modelMap.size) {
        resultsMap += (i -> predictPoint(modelMap.get(i).orNull, point.features))
      }
      (point.label, Math.round(resultsMap.maxBy(_._2)._1))
    })
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / labelAndPreds.count
    trainErr
  }


  def findErrorRate(labelledRDD: RDD[LabeledPoint], model: LogisticRegressionModel): Double = {
    val broadcastModel = labelledRDD.context.broadcast(model)
    val labelAndPreds = labelledRDD.map {
      point =>
        val model = broadcastModel.value
        val prediction = model.predict(point.features)
        (point.label, prediction)
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / labelAndPreds.count
    trainErr
  }

  def predictPoint(model: LogisticRegressionModel, features: Array[Double]): Double = {
    val testDataMat = new DoubleMatrix(1, features.length, features: _*)
    val weightMatrix = new DoubleMatrix(model.weights.length, 1, model.weights: _*)
    val margin = testDataMat.mmul(weightMatrix).get(0) + model.intercept
    val value = 1.0 / (1.0 + math.exp(margin * -1))
    value
  }

  def doPredictionMultiClass(modelMap: Map[Int, LogisticRegressionModel], features: Array[Double], featureMean: Array[Double] = null, featureStd: Array[Double] = null): Double = {
    if (featureMean != null && featureStd != null) {
      val normalizedFeatures = new Array[Double](features.length)
      for (a <- 0 to features.length - 1) {
        normalizedFeatures(a) = (features(a) - featureMean(a)) / featureStd(a)
      }
      var resultsMap: Map[Int, Double] = Map()
      for (i <- 1 to modelMap.size) {
        resultsMap += (i -> predictPoint(modelMap.get(i).orNull, normalizedFeatures))
      }
      resultsMap.maxBy(_._2)._1

    } else {
      var resultsMap: Map[Int, Double] = Map()
      for (i <- 1 to modelMap.size) {
        resultsMap += (i -> predictPoint(modelMap.get(i).orNull, features))
      }
      resultsMap.maxBy(_._2)._1
    }
  }

  def doPrediction(model: LogisticRegressionModel, features: Array[Double], featureMean: Array[Double] = null, featureStd: Array[Double] = null): Double = {
    if (featureMean != null && featureStd != null) {
      val normalizedFeatures = new Array[Double](features.length)
      for (a <- 0 to features.length - 1) {
        normalizedFeatures(a) = (features(a) - featureMean(a)) / featureStd(a)
      }
      val finalPredictedValue = model.predict(normalizedFeatures)
      finalPredictedValue
    } else {
      val finalPredictedValue = model.predict(features)
      finalPredictedValue
    }
  }

  def runRegression(labelledRDD: RDD[LabeledPoint], numberOfIterations: Int, initialWeights: Array[Double], stepSize: Double, miniBatchFraction: Double): LogisticRegressionModel = {
    val regression = new LogisticRegressionWithSGD()
    regression.optimizer.setNumIterations(numberOfIterations).setStepSize(stepSize).setMiniBatchFraction(miniBatchFraction)
    val logisticRegressionModel = regression.run(labelledRDD, initialWeights)
    logisticRegressionModel
  }

  def polynomialFeatures(featureArray: Array[Double], degree: Int): Array[Double] = {
    var outputArray = Array[Double]()
    for (i <- 1 to degree; j <- 0 to i) {
      outputArray +:= scala.math.pow(featureArray(0), i - j) * scala.math.pow(featureArray(1), j)
    }
    outputArray
  }

  def runRegularizedRegression(labelledRDD: RDD[LabeledPoint], numberOfIterations: Int, initialWeights: Array[Double], stepSize: Double, miniBatchFraction: Double, regParam: Double): LogisticRegressionModel = {
    val regression = new LogisticRegressionWithSGD()
    regression.optimizer.setNumIterations(numberOfIterations).setStepSize(stepSize).setMiniBatchFraction(miniBatchFraction).setRegParam(regParam).setUpdater(new L1Updater)
    val logisticRegressionModel = regression.run(labelledRDD, initialWeights)
    logisticRegressionModel
  }

  def normaliseFeatures(labelledRDD: RDD[LabeledPoint]): (RDD[LabeledPoint], Array[Double], Array[Double]) = {
    val numOfFeatures = labelledRDD.first().features.length
    val results = MLUtils.computeStats(labelledRDD, numOfFeatures, labelledRDD.count())
    val featureMean = results._2.toArray
    val featureStdDev = results._3.toArray
    val broadcastMeanAndStdDev = labelledRDD.context.broadcast(featureMean,featureStdDev)

    val normalizedRDD = labelledRDD.map(point => {
      val normalizedFeatureArray = new Array[Double](numOfFeatures)
      val features = point.features
      for (a <- 0 to numOfFeatures - 1) {
        if (broadcastMeanAndStdDev.value._1(a) == 0 && broadcastMeanAndStdDev.value._2(a) == 0) {
          normalizedFeatureArray(a) = 0.0
        } else {
          normalizedFeatureArray(a) = (features(a) - broadcastMeanAndStdDev.value._1(a)) / broadcastMeanAndStdDev.value._2(a)
        }
      }
      LabeledPoint(point.label, normalizedFeatureArray)
    })
    (normalizedRDD, featureMean.toArray, featureStdDev.toArray)
  }

  def split[T: Manifest](data: RDD[T], percentage: Double, seed: Long = System.currentTimeMillis()): (RDD[T], RDD[T]) = {
    val randomNumberGenForPartitions = new Random(seed)
    val partitionRandomNumber = data.partitions.map(each => randomNumberGenForPartitions.nextLong())
    val temp = data.mapPartitionsWithIndex((index, iterator) => {
      val randomNumberGenForRows = new Random(partitionRandomNumber(index))
      val intermediate = iterator.map(each => {
        (each, randomNumberGenForRows.nextDouble())
      })
      intermediate
    })
    (temp.filter(_._2 <= percentage).map(_._1), temp.filter(_._2 > percentage).map(_._1))
  }
}
