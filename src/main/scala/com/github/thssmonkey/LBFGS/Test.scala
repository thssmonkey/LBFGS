package com.github.thssmonkey.LBFGS

import org.apache.flink.api.scala._
import org.apache.flink.ml.common._
import org.apache.flink.ml.math._

/**
  * 1. import LBFGS相关API
  */
import com.github.thssmonkey.LBFGS._

/**
  * Test Flink's LBFGS for data analysis using case using the default of credit card clients dataset from UCI
  * @see http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
  *
  * args(0) should be the local path of the dataset.
  *
  */
object Test extends App{
  val env = ExecutionEnvironment.getExecutionEnvironment

  val m = 50
  val n = 5000
  val test_num = 200
  val var_num = 20

  val data = env.readCsvFile[(String, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double)](args(0), ignoreFirstLine = true)
    .filter(_._1.replace("\"", "").toInt <= (n + test_num))

  val toLabeledVector = { (t: (String, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double, Double)) =>
    val features = t match {
      case (_, limit_bal, age, pay1, pay2, pay3, pay4, pay5, pay6, bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6, pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6, _)
      => VectorBuilder.vectorBuilder.build(limit_bal :: age :: pay1 :: pay2 :: pay3 :: pay4 :: pay5 :: pay6 :: bill_amt1 :: bill_amt2 :: bill_amt3 :: bill_amt4 :: bill_amt5 :: bill_amt6 :: pay_amt1 :: pay_amt2 :: pay_amt3 :: pay_amt4 :: pay_amt5 :: pay_amt6 :: Nil )
    }
    LabeledVector(t._22, features)
  }

  val training = data.filter(_._1.replace("\"", "").toInt <= n).map(toLabeledVector)
  val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

  /**
    * 2. 创建LBFGS实例，并设置参数
    */
  val lbfgs = LBFGS()
    .setLossFunction(lossFunction)
    .setIterations(1000)
    //.setStepsize(0.0001)
    .setConvergenceThreshold(0.001)
    .setStorages(m)

  val initialWeights = Some(DenseVector.zeros(var_num))

  /**
    * 3. 运行得到结果
    */

  val startTime = System.currentTimeMillis()
  for( i <- 1 to 10) {
    lbfgs.optimize(training, initialWeights)
  }
  val endTime = System.currentTimeMillis()
  println("time: " + (endTime - startTime) / 10)
  val weights = lbfgs.optimize(training, initialWeights)
  println(weights)

  val test = data.filter(_._1.replace("\"", "").toInt > n).map(toLabeledVector)
  test.map { l => (l, weights) }
      .map(x => (x._1.vector.dot(x._2), x._1.label, x._2))
      .map(x =>(LBFGSSquaredLoss.loss(x._1, x._2) / test_num, x._3))
      .sum(0)
      .print()
}


/**
  * Advertising result:
  * DenseVector(0.05488324053036032, 0.2156819005580397, 0.016836151594152398)
  * (1.8643101949719485,DenseVector(0.05488324053036032, 0.2156819005580397, 0.016836151594152398))
  *
  * Credit result:
  * DenseVector(-7.235584526558495, 0.23097356001003377, 0.3275784947343272, 0.8976121951905398, -2.8406550839945215, -14.795797429783507)
  * (13172.506834068485,DenseVector(-7.235584526558495, 0.23097356001003377, 0.3275784947343272, 0.8976121951905398, -2.8406550839945215, -14.795797429783507))
  *
  * Spark
  * [-7.235586112399782,0.2309735104729321,0.3275793466781157,0.8976127335115671,-2.8406550055803046,-14.79579459239586]
  * (13172.506773430216,[-7.235586112399782,0.2309735104729321,0.3275793466781157,0.8976127335115671,-2.8406550055803046,-14.79579459239586])
  *
  */

