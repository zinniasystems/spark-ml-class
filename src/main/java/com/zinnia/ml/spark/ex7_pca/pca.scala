package com.zinnia.pca

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.jblas.{DoubleMatrix}
import scala.util.control.Breaks
import scala.Array

/**
 * Created with IntelliJ IDEA.
 * User: Shashi
 * Date: 29/1/14
 * Time: 3:16 PM
 * To change this template use File | Settings | File Templates.
 */

/*Class to calculate PCA of the given data*/

class pca extends Serializable {
  var columnMean = Array[Double]()
  var columnStd = Array[Double]()
  var normalisedColumnMean = Array[Double]()
  var normalisedColumnStd = Array[Double]()
  var extractedUData = Array[Array[Double]]()

  /*Method to find the covariance matrix for the given RDD*/
  def getCovarianceMatrix(input:RDD[Array[Double]]):DoubleMatrix={
    val columnsCount : Int = input.first().length.toInt
    var tempArray = Array[String]()
    for (i <- 0 to normalisedColumnMean.length-1){
      for (j <- i+1 to normalisedColumnMean.length-1) {
        var rows = i+","+normalisedColumnMean(i).toString+","+j+","+normalisedColumnMean(j).toString
//        println(rows)
        tempArray :+= rows
      }
    }
    val pairedMeanTuple = tempArray.map(data=>{
      val parts = data.split(",")
      (parts(0).toInt,parts(1).toDouble,parts(2).toInt,parts(3).toDouble)
    })

   val varianceTuple =  pairedMeanTuple.map(item=>{
      val output =  input.map(elements=>{
        (elements(item._1)-item._2)*(elements(item._3)-item._4)
       }).reduce((a,b)=>a+b)
      ((item._1,item._3),output)
    }).map(pairs=>{
     (pairs._1,pairs._2/(input.count()-1))
   })

    var varianceTupleList = List[((Int,Int),Double)]()
    varianceTupleList = varianceTuple.toList
    var matrix = Array[Array[Double]]()
    for (i <- 0 to columnsCount-1){
      var rowArray = Array[Double]()
      for(j<-0 to columnsCount-1){
        if(i==j){
          rowArray:+=scala.math.pow(normalisedColumnStd(i),2)
        }else{
          varianceTupleList.foreach(ele=>{
            val index = ele._1
            if(index._1==i && index._2==j||index._1==j && index._2==i)rowArray:+=ele._2
          })
        }
      }
//           println(rowArray.toList.toString())
      matrix:+=rowArray
    }
    new DoubleMatrix(matrix)
  }

  /*Method to find the column-wise Mean and Std-Deviation of the RDD*/

  def computeMeanStd(data:RDD[Array[Double]]):(Array[Double], Array[Double])={
    val nexamples = data.count()
    var meanArray = Array[Double]()
    var stdArray = Array[Double]()
    val xColSumSq: RDD[(Int,(Double,Double))] = data.flatMap ({ rowArray =>
      val nCols = rowArray.length
      Iterator.tabulate(nCols) { i =>
        (i, (rowArray(i), rowArray(i)*rowArray(i)))
      }
    })
    val output =  xColSumSq.reduceByKey((x1,x2)=>(x1._1+x2._1,x1._2+x2._2))
    val stats = output.toArray()

    stats.foreach(ele=>{
      val tuple = ele._2
      meanArray:+=tuple._1/nexamples
      val variance = (tuple._2-(Math.pow(tuple._1,2)/nexamples))/(nexamples)
      stdArray :+= math.sqrt(variance)
    })

    (meanArray,stdArray)
  }

  /*Method to perform feature scaling*/

  def getNormalisedRDD(input:RDD[Array[Double]]):RDD[Array[Double]]={
    input.map(line=>{
      var doubleArray = Array[Double]()
      for(i <- Range(0,line.length)){
        doubleArray :+= line(i)-columnMean(i)/columnStd(i)
      }
      doubleArray
    })
  }

  /* Method to find K (Count) of Principal components of the input n features*/

  def getKValue(SMatrix:Array[Double]):Int={
    val loop = new Breaks
    var K : Int = 0
    var arraySum : Double = 0.0
    SMatrix.foreach(arraySum+=_)
    loop.breakable{for(i<- 0 to SMatrix.length-1){
      var coefficient : Double = 0.0
      var numerator : Double = 0.0
      for (j<-0 to i){
        numerator = numerator + SMatrix(j)
      }
      coefficient = 1-(numerator/arraySum)
      if(coefficient<=0.01){
        K = i+1
        loop.break
      }
    } }
    K
  }

  /*Method to parse the data*/

  def parseData(data:RDD[String]):RDD[Array[Double]]={
    val inputRDD = data.map(line=>{
      val elements = line.split(",")
      var output = Array[Double]()
      elements.foreach(ele=>{
        output:+=ele.toDouble
      })
      output
    })
    inputRDD
  }

  /*Method to extract the K rows from the U Matrix*/

  def extractUData(K:Int,UMatrix:DoubleMatrix):Array[Array[Double]]={
    var extractedUData = Array[Array[Double]]()

    for(i <- 0 to K-1){
      extractedUData :+= UMatrix.getColumn(i).toArray
    }
    extractedUData
  }

  /*Method to project the data*/

  def projectData(normalisedRDD:RDD[Array[Double]]):Array[Array[Double]]={
    val extractedU = new DoubleMatrix(extractedUData).toArray2
    var projectedArrays = Array[Array[Double]]()
    extractedU.foreach(eigenVector=>{
      var outputArray = Array[Double]()
      normalisedRDD.toArray().toList.foreach(ele=>{
        var sum : Double = 0.0
        for(i<-Range(0,ele.length)){
          sum = sum + eigenVector(i)*ele(i)
        }
        outputArray :+= sum
      })
      projectedArrays :+= outputArray
    })

    projectedArrays
  }
}

