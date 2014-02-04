package com.zinnia.ml.spark.util

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.{LinearRegressionModel, LinearRegressionWithSGD, LabeledPoint}
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionModel}


/**
 * Created with IntelliJ IDEA.
 * User: hadoop
 * Date: 20/1/14
 * Time: 4:13 PM
 * To change this template use File | Settings | File Templates.
 */

object RegressionUtil {

  //parses a csv file where each line has a set of features at the beginning and the label at the end
  def parseFileContent(inputData: RDD[String]): RDD[LabeledPoint] = {
    val labelledRDD = inputData.map(line => {
      val parts = line.split(",")
      LabeledPoint(parts.last.toDouble, parts.init.map(x => x.toDouble).toArray)
    })
    labelledRDD
  }

}



