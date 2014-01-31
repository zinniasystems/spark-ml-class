package com.zinnia.ml.spark.ex2_logistic_regression

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import com.zinnia.ml.spark.util.RegressionUtil
import scala.collection.mutable
import org.jblas.DoubleMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext
import java.lang.Math
import org.apache.spark.mllib.optimization.{SquaredL2Updater, L1Updater}
import java.util.Random

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 21/1/14
 * Time: 11:26 PM
 * To change this template use File | Settings | File Templates.
 */
class LogisticRegression {

  def runMultiClassRegression(listOfMappedRDDs : List[(Int,RDD[LabeledPoint])], numberOfIterations: Int, initialWeights:Array[Double], stepSize:Double,miniBatchFraction:Double):Map[Int,LogisticRegressionModel] ={
    //var modelMap : mutable.Map[Int,LogisticRegressionModel] = mutable.Map()
    val rddOfModels = listOfMappedRDDs.map(eachRDD=>{
      val regression = new LogisticRegressionWithSGD()
      regression.optimizer.setNumIterations(numberOfIterations).setStepSize(stepSize).setMiniBatchFraction(miniBatchFraction)
      val model = regression.run(eachRDD._2,initialWeights)
      (eachRDD._1,model)
    })
    rddOfModels.toMap

  }



  def findErrorRateMultiClass(modelMap:Map[Int,LogisticRegressionModel],labelledRDD: RDD[LabeledPoint]):Double = {
    val labelAndPreds = labelledRDD.map( point => {
      def predictPoint(model:LogisticRegressionModel, testData:Array[Double]):Double={
        val testDataMat = new DoubleMatrix(1, testData.length, testData:_*)
        val weightMatrix = new DoubleMatrix(model.weights.length, 1, model.weights:_*)
        val margin = testDataMat.mmul(weightMatrix).get(0) + model.intercept
        val value = 1.0/ (1.0 + math.exp(margin * -1))
        value
      }
      var resultsMap:Map[Int,Double] = Map()
      for(i <- 1 to modelMap.size){
        resultsMap += (i -> predictPoint(modelMap.get(i).orNull, point.features))
      }
      (point.label,Math.round(resultsMap.maxBy(_._2)._1))
    } )
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / labelAndPreds.count
    trainErr
  }


  def findErrorRate(labelledRDD: RDD[LabeledPoint], model: LogisticRegressionModel): Double = {
    val labelAndPreds = labelledRDD.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / labelAndPreds.count
    trainErr
  }
  

  def predictPoint(model:LogisticRegressionModel, features:Array[Double]):Double={
    val testDataMat = new DoubleMatrix(1, features.length, features:_*)
    val weightMatrix = new DoubleMatrix(model.weights.length, 1, model.weights:_*)
    val margin = testDataMat.mmul(weightMatrix).get(0) + model.intercept
    val value = 1.0/ (1.0 + math.exp(margin * -1))
    value
  }

  def doPredictionMultiClass(modelMap:Map[Int,LogisticRegressionModel], features:Array[Double],featureMeanStd:Array[(Double,Double)] = null):Double={
    if (featureMeanStd != null) {
      val normalizedFeatures = new Array[Double](features.length)
      for (a <- 0 to features.length - 1) {
        normalizedFeatures(a) = (features(a) - featureMeanStd(a)._1) / featureMeanStd(a)._2
      }
      var resultsMap:Map[Int,Double] = Map()
      for(i <- 1 to modelMap.size){
        resultsMap += (i -> predictPoint(modelMap.get(i).orNull, normalizedFeatures))
      }
      resultsMap.maxBy(_._2)._1

    } else {
      var resultsMap:Map[Int,Double] = Map()
      for(i <- 1 to modelMap.size){
        resultsMap += (i -> predictPoint(modelMap.get(i).orNull, features))
      }
      val finalPredictedValue = resultsMap.maxBy(_._2)._1
      finalPredictedValue
    }

  }



  def doPredictPoint(model:LogisticRegressionModel, features:Array[Double],featureMeanStd:Array[(Double,Double)] = null):Double={
    if (featureMeanStd != null) {
      val normalizedFeatures = new Array[Double](features.length)
      for (a <- 0 to features.length - 1) {
        normalizedFeatures(a) = (features(a) - featureMeanStd(a)._1) / featureMeanStd(a)._2
      }
      val finalPredictedValue = predictPoint(model,normalizedFeatures)
      finalPredictedValue
    } else {
      val finalPredictedValue = predictPoint(model,features)
      finalPredictedValue
    }
  }


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



  def runRegression(labelledRDD: RDD[LabeledPoint], numberOfIterations: Int, initialWeights:Array[Double], stepSize:Double,miniBatchFraction:Double): LogisticRegressionModel = {
    val regression = new LogisticRegressionWithSGD()
    regression.optimizer.setNumIterations(numberOfIterations).setStepSize(stepSize).setMiniBatchFraction(miniBatchFraction)
    val logisticRegressionModel = regression.run(labelledRDD,initialWeights)
    logisticRegressionModel
  }

  def mapfeatures(featureArray:Array[Double],degree:Int):Array[Double] = {
    var outputArray = Array[Double]()
    for (i <- 1 to degree ;j<-0 to i){
      //      println(i+","+j)
      outputArray +:= scala.math.pow(featureArray(0),i-j)*scala.math.pow(featureArray(1),j)
    }
    outputArray
  }

  def runRegularizedRegression(labelledRDD: RDD[LabeledPoint], numberOfIterations: Int, initialWeights:Array[Double], stepSize:Double,miniBatchFraction:Double,regParam:Double): LogisticRegressionModel = {
    val regression = new LogisticRegressionWithSGD()
    regression.optimizer.setNumIterations(numberOfIterations).setStepSize(stepSize).setMiniBatchFraction(miniBatchFraction).setRegParam(regParam).setUpdater(new L1Updater)
    val logisticRegressionModel = regression.run(labelledRDD,initialWeights)
    logisticRegressionModel
  }

  def normaliseFeatures(labelledRDD: RDD[LabeledPoint]): (RDD[LabeledPoint],Array[(Double,Double)]) = {
    val numOfFeatures = labelledRDD.first().features.length
    println(numOfFeatures)
    val featureMeanAndStdDev = new Array[(Double,Double)](numOfFeatures)
    for (a <- 0 to labelledRDD.first().features.length-1){
      val singleFeature = labelledRDD.map(point => {
        point.features(a)
      })
      singleFeature.cache()
      featureMeanAndStdDev(a) = (RegressionUtil.calcMeanAndStdDev(singleFeature.toArray()))
    }

    val normalizedRDD = labelledRDD.map(point => {
      val normalizedFeatureArray = new Array[Double](numOfFeatures)
      val features = point.features
      for (a <- 0 to numOfFeatures-1) {
        if(featureMeanAndStdDev(a)._1 == 0 && featureMeanAndStdDev(a)._2 == 0){
          normalizedFeatureArray(a) = 0.0
        } else {
          normalizedFeatureArray(a) = (features(a) - featureMeanAndStdDev(a)._1) / featureMeanAndStdDev(a)._2
        }
      }
      LabeledPoint(point.label, normalizedFeatureArray)
    })
    (normalizedRDD,featureMeanAndStdDev)

  }

  def split[T:Manifest](data:RDD[T], percentage:Double, seed:Long = System.currentTimeMillis()):(RDD[T],RDD[T])={
    val randomNumberGenForPartitions = new Random(seed)
    val partitionRandomNumber = data.partitions.map(each=>randomNumberGenForPartitions.nextLong())
    val temp = data.mapPartitionsWithIndex((index,iterator)=>{
      val randomNumberGenForRows = new Random(partitionRandomNumber(index))
      val intermediate = iterator.map(each=>{
        (each,randomNumberGenForRows.nextDouble())
      })
      intermediate
    })
    (temp.filter(_._2 <= percentage).map(_._1),temp.filter(_._2 > percentage).map(_._1))
  }

}
