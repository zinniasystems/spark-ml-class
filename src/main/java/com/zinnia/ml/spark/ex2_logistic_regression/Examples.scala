package com.zinnia.ml.spark.ex2_logistic_regression

import org.apache.spark.SparkContext
import com.zinnia.ml.spark.util.RegressionUtil
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD

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
    val model = regression.runRegression(labelledRDD,10,Array(0.0,0.0),0.5,1.0)
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
    val model = regression.runRegression(labelledRDD,10,Array(0.0,0.0),1.0,1.0)
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

object Ex2_Data2_Poly{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local", "ml-exercise")
    val fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex2_logistic_regression/ex2data2.txt").cache()
    println("Running Fourth Examples : ex2_logistic_regression Regression 2")
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Running Regression")
    println("Number of features is "+labelledRDD.first().features.length)
    println("Count is "+labelledRDD.count())
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
    println("Number of features is "+labelledRDD.first().features.length)
    println("Count is "+labelledRDD.count())
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMeanAndStdDev = featureScaledData._2
    val thetaArray = new Array[Double](labelledRDD.first().features.length)
    //val model = regression.runRegression(labelledRDD,20,thetaArray,1.0,1.0)
    val model = regression.runRegularizedRegression(labelledRDD,20,thetaArray,1.0,1.0,0.05)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate is:" + errorRate)
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)
    println("Prediction for the values -0.21947,-0.016813 :")
    println(regression.doPrediction(model,regression.polynomialFeatures(Array(-0.21947,-0.016813),degreeOfPoly),featureMeanAndStdDev))
    println("Prediction for the values 0.60426,0.59722 :")
    println(regression.doPrediction(model,regression.polynomialFeatures(Array(0.60426,0.59722),degreeOfPoly),featureMeanAndStdDev))

  }
}

object Ex3_Data3{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local","ml-exercise")
    val fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex2_logistic_regression/ex2data3.txt")
    println("Runnin multiclass logistic regression")
    println("Parsing file contents")
    val labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Normalizing the features")
    val labelledRDDData = regression.normaliseFeatures(labelledRDD)
    val normalizedLabelledRDD = labelledRDDData._1.cache()
    val featureMeanAndStdDev = labelledRDDData._2
    println("Creating multiple RDDs out of single RDD")
    val multipleDataRDDs = normalizedLabelledRDD.flatMap(eachPoint=>{
      var labeledPointArray = new Array[(Int,LabeledPoint)](0)
      for(a <- 1 to 4){
        var label = 0.0
        if(eachPoint.label == a){
          label = 1.0
        }
        labeledPointArray = labeledPointArray :+ (a,LabeledPoint(label,eachPoint.features))
      }
      labeledPointArray
    })
    var listOfMappedRDDs : List[(Int,RDD[LabeledPoint])] = Nil
    println("Groupin RDD s based on key")
    for(i <- 1 to 4){
      val eachMappedRDD = multipleDataRDDs.filter(eachRDD => if(eachRDD._1 == i) true else false).map(eachFilteredRDD=>eachFilteredRDD._2)
      listOfMappedRDDs = listOfMappedRDDs.::(i,eachMappedRDD)
    }
    println("Total number of RDDs " + multipleDataRDDs.count())

    val thetaArray = new Array[Double](4)
    println("Running regression on RDDs")
    val modelMap = regression.runMultiClassRegression(listOfMappedRDDs,10,thetaArray,0.01,1.0)
    println("Number of models "+modelMap.size)

    val errorRate = regression.findErrorRateMultiClass(modelMap,normalizedLabelledRDD)
    println(errorRate)

    println("Error rate is "+errorRate )
//    val result = regression.doPredictionMultiClass(modelMap,Array(5.6,3.0,4.1,1.3),featureMeanAndStdDev)
//    println("Result for : 6.7,3.1,5.6,2.4 " + result)

  }
}




object Ex3_Data4{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local","ml-exercise")
    val fileContentsMain = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex2_logistic_regression/multiclass.csv").cache()
    val fileContents = context.makeRDD(fileContentsMain.take(100)).cache()
    println("Parsing file contents")
    val labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Normalizing the features")
    val labelledRDDData = regression.normaliseFeatures(labelledRDD)
    val normalizedLabelledRDD = labelledRDDData._1.cache()
    val featureMeanAndStdDev = labelledRDDData._2
    println("Creating multiple RDDs out of single RDD")
    val multipleDataRDDs = normalizedLabelledRDD.flatMap(eachPoint=>{
      var labeledPointArray = new Array[(Int,LabeledPoint)](0)
      for(a <- 1 to 10){
        var label = 0.0
        if(eachPoint.label == a){
          label = 1.0
        }
        labeledPointArray = labeledPointArray :+ (a,LabeledPoint(label,eachPoint.features))
      }
      labeledPointArray
    })


    var listOfMappedRDDs : List[(Int,RDD[LabeledPoint])] = Nil
    println("Groupin RDD s based on key")
    for(i <- 1 to 10){
      val eachMappedRDD = multipleDataRDDs.filter(eachRDD => if(eachRDD._1 == i) true else false).map(eachFilteredRDD=>eachFilteredRDD._2)
      listOfMappedRDDs = listOfMappedRDDs.::(i,eachMappedRDD)
    }
    println("Total number of RDDs " + multipleDataRDDs.count())

    val thetaArray = new Array[Double](400)
    println("Running regression on RDDs")
    val modelMap = regression.runMultiClassRegression(listOfMappedRDDs,10,thetaArray,0.01,1.0)
    println("Number of models "+modelMap.size)

    val errorRate = regression.findErrorRateMultiClass(modelMap,normalizedLabelledRDD)
    println(errorRate)

  }
}


object Ex3_RealTimeData{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local", "ml-exercise")
    var fileContents = context.textFile("/home/hadoop/MachineLearning/Exp/MachineLearningExperiment/src/main/resources/ex2_logistic_regression/RealTimeDataLR.csv").cache()
    println("Running RealTime Data Experiment : ex2_logistic_regression Regression")
    println("Reading File")
    var labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    val splittedRDDs = regression.split(labelledRDD,0.70)
    labelledRDD = splittedRDDs._1
    var testingRDD = splittedRDDs._2
    println("Running Regression")
    val featureScaledData = regression.normaliseFeatures(labelledRDD)
    labelledRDD = featureScaledData._1.cache()
    val featureMeanAndStdDev = featureScaledData._2
    val thetaArray = new Array[Double](10)
    //val model = regression.runRegression(labelledRDD,10,thetaArray,1.0,1.0)
    val model = regression.runRegularizedRegression(labelledRDD,10,thetaArray,1.0,1.0,0.01)
    println("Finding Error Rate")
    val errorRate = regression.findErrorRate(labelledRDD, model)
    println("Error Rate for 70% of the data is:" + errorRate*100+"%")

    val testDataFeatureScaled = regression.normaliseFeatures(testingRDD)
    testingRDD = testDataFeatureScaled._1
    val testDataErrorRate = regression.findErrorRate(testingRDD,model)
    println("Error rate for remaining 30% data when predicted is: "+testDataErrorRate*100+"%")
    println("Theta values are:")
    println(model.intercept)
    model.weights.foreach(println)

  }
}
