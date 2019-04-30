package com.github.thssmonkey.LBFGS

import org.apache.flink.api.scala._
import org.apache.flink.ml.common._
import org.apache.flink.ml.math._
import org.apache.flink.ml.optimization.SquaredLoss

/**
  * Test Flink's LBFGS for a simple linear regression use case using the advertisement dataset from ISL
  * @see https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv
  *
  * args(0) should be the local path of the dataset.
  */
object example extends App{
  val env  = ExecutionEnvironment.getExecutionEnvironment

  val data = env.readCsvFile[(String, Double, Double, Double, Double)](args(0), ignoreFirstLine = true)

  val toLabeledVector = { (t: (String, Double, Double, Double, Double)) =>
    val features = t match {
      case (_, tv, radio, newspaper, _)
      => VectorBuilder.vectorBuilder.build(tv :: radio :: newspaper :: Nil )
    }
    LabeledVector(t._5, features)
  }

  val training = data.filter(_._1.replace("\"", "").toInt <= 150).map(toLabeledVector)
  val test = data.filter(_._1.replace("\"", "").toInt > 150).map(toLabeledVector)
  val lossFunction = LBFGSGenericLossFunction(SquaredLoss, LinearPrediction)

  //training.print()
  val lbfgs = LBFGS()
    .setLossFunction(lossFunction)
    .setIterations(1000)
    //.setStepsize(0.0001)
    .setConvergenceThreshold(1e-9)      //(0.001)
    .setStorages(10)

  val initialWeights = Some(DenseVector.zeros(3).asInstanceOf[Vector])
  val weights = lbfgs.optimize(training, initialWeights)
  println(weights)

  test.map {
    l => (l, weights)
  }.map(x => (x._1.vector.dot(x._2), x._1.label, x._2))
    .map(x =>(SquaredLoss.loss(x._1, x._2) / 50, x._3))
    .sum(0)
    .print()
}


/**
  * val lossFunction = GenericLossFunction(SquaredLoss, LinearPrediction)
  *
  * //training.print()
  * val lbfgs = LBFGS()
  * .setLossFunction(lossFunction)
  * .setIterations(1000)
  * //.setStepsize(0.0001)
  * .setConvergenceThreshold(0.001)
  * .setStorages(10)
  */

/**
  * SGD result:
  * WeightVector(DenseVector(0.060404044749571666, 0.11677110922434246, 0.06059362625506299),0.0030875096770564417)
  * (3.6879336800379248,WeightVector(DenseVector(0.060404044749571666, 0.11677110922434243, 0.06059362625506301),0.0030875096770564417))
  */

/**
  * FLINK result:
  * DenseVector(0.05488324053036032, 0.2156819005580397, 0.016836151594152398)
  * (1.8643101949719485,DenseVector(0.05488324053036032, 0.2156819005580397, 0.016836151594152398))
  */

/**
  * SPARK result:
  * [0.05488324053036032,0.2156819005580397,0.0168361515941524]
  * (1.8643101949719487,[0.05488324053036032,0.2156819005580397,0.0168361515941524])
  */
