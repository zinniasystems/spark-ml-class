package com.zinnia.ml.spark.ex3_multiclass_classification

import com.zinnia.ml.spark.ex2_logistic_regression.LogisticRegression
import org.apache.spark.SparkContext
import com.zinnia.ml.spark.util.RegressionUtil
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 4/2/14
 * Time: 12:46 PM
 * To change this template use File | Settings | File Templates.
 */
object Ex3_Data4{
  def main(args: Array[String]) {
    val regression = new LogisticRegression
    val context = new SparkContext("local","ml-exercise")
    val fileContents = context.textFile("src/main/resources/ex2_logistic_regression/ex3data1.csv").cache()
    println("Parsing file contents")
    val labelledRDD = RegressionUtil.parseFileContent(fileContents).cache()
    println("Normalizing the features")
    val labelledRDDData = regression.normaliseFeatures(labelledRDD)
    val normalizedLabelledRDD = labelledRDDData._1.cache()
    println("Creating multiple RDDs out of single RDD")

    //Generate a RDD of label and labelled point object as a tuple, this would generate a RDD of size number_of_labels*number_of_examples

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
      //filter the RDD created in the above step into multiple RDDs based on the first value of tuple.
      val eachMappedRDD = multipleDataRDDs.filter(eachRDD => if(eachRDD._1 == i) true else false).map(eachFilteredRDD=>eachFilteredRDD._2)
      listOfMappedRDDs = listOfMappedRDDs.::(i,eachMappedRDD)
    }

    val thetaArray = new Array[Double](400)
    println("Running regression on RDDs")
    val modelMap = regression.runMultiClassRegression(listOfMappedRDDs,50,thetaArray,1,1.0)
    println("Number of models "+modelMap.size)
    println("Finding error rate")
    val errorRate = regression.computeCostMultiClass(modelMap,normalizedLabelledRDD)
    println("Error rate is:"+errorRate)

  }
}

