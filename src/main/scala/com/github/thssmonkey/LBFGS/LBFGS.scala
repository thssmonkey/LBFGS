/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.thssmonkey.LBFGS

import org.apache.flink.api.scala._
import org.apache.flink.ml.common._
import org.apache.flink.ml.math._
import org.apache.flink.ml.math.Breeze._
import org.apache.flink.ml.optimization.RegularizationPenalty
import com.github.thssmonkey.LBFGS.LBFGSIterativeSolver._
import com.github.thssmonkey.LBFGS.LBFGSSolver._
import com.github.thssmonkey.LBFGS.LBFGSLearningRateMethod.LBFGSLearningRateMethodTrait

import scala.collection.mutable
import breeze.linalg.{DenseVector => BreezeDenseVector}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}

/** Base class which performs Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimization.
  *
  * For each labeled vector in a mini batch the gradient is computed and added to a partial
  * gradient. The partial gradients are then summed and divided by the size of the batches. The
  * average gradient is then used to updated the weight values, including regularization.
  *
  * At the moment, the whole partition is used for L-BFGS, making it effectively a batch gradient
  * descent. Once a sampling operator has been introduced, the algorithm can be optimized
  *
  *  The parameters to tune the algorithm are:
  *                      [[LBFGSSolver.LBFGSLossFunction]] for the loss function to be used,
  *                      [[LBFGSSolver.RegularizationPenaltyValue]] for the regularization penalty.
  *                      [[LBFGSSolver.RegularizationConstant]] for the regularization parameter,
  *                      [[LBFGSIterativeSolver.Storages]] for the number of corrections,
  *                      [[LBFGSIterativeSolver.Iterations]] for the maximum number of iteration,
  *                      [[LBFGSIterativeSolver.LearningRate]] for the learning rate used.
  *                      [[LBFGSIterativeSolver.ConvergenceThreshold]] when provided the algorithm will
  *                      stop the iterations if the relative change in the value of the objective
  *                      function between successive iterations is is smaller than this value.
  *                      [[LBFGSIterativeSolver.LearningRateMethodValue]] determines functional form of
  *                      effective learning rate.
  */
class LBFGS extends LBFGSIterativeSolver {

  /** Provides a solution for the given optimization problem
    *
    * @param data           A Dataset of LabeledVector (label, features) pairs
    * @param initialWeights The initial weights that will be optimized
    * @return The weights, optimized for the provided data.
    */
  override def optimize(
                         data: DataSet[LabeledVector],
                         initialWeights: Option[Vector]): Vector = {

    val numberOfStorages: Int = parameters(Storages)
    val numberOfIterations: Int = parameters(Iterations)
    val convergenceThreshold: Double = parameters(ConvergenceThreshold)
    val lossFunction = parameters(LBFGSLossFunction)
    val learningRate = parameters(LearningRate)
    val regularizationPenalty = parameters(RegularizationPenaltyValue)
    val regularizationConstant = parameters(RegularizationConstant)
    val learningRateMethod = parameters(LearningRateMethodValue)

    // Initialize weights
    val newInitialWeights = createInitialWeights(initialWeights, data)

    // Perform the iterations
    optimizeWithIterations(
      data,
      newInitialWeights,
      numberOfStorages,
      numberOfIterations,
      regularizationPenalty,
      regularizationConstant,
      learningRate,
      convergenceThreshold,
      lossFunction,
      learningRateMethod)
  }

  def optimizeWithIterations(
                              dataPoints: DataSet[LabeledVector],
                              initialWeights: Vector,
                              numberOfStorages: Int,
                              numberOfIterations: Int,
                              regularizationPenalty: RegularizationPenalty,
                              regularizationConstant: Double,
                              learningRate: Double,
                              convergenceThreshold: Double,
                              lossFunction: LBFGSLossFunction,
                              learningRateMethod: LBFGSLearningRateMethodTrait)
  : Vector = {
    val lossHistory = mutable.ArrayBuilder.make[Double]
    val numberOfExamples = dataPoints.count()
    val costDiffFun = new costDiffFunction(dataPoints, lossFunction, regularizationPenalty, regularizationConstant, learningRate, learningRateMethod, numberOfExamples)
    val lbfgs = new BreezeLBFGS[BreezeDenseVector[Double]](numberOfIterations, numberOfStorages, convergenceThreshold)
    val states = lbfgs.iterations(new CachedDiffFunction(costDiffFun), initialWeights.asBreeze.toDenseVector)
    var state = states.next()
    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }
    lossHistory += state.value
    val weights = state.x.fromBreeze
    weights
  }

  private class costDiffFunction(
                                  dataPoints: DataSet[LabeledVector],
                                  lossFunction: LBFGSLossFunction,
                                  regularizationPenalty: RegularizationPenalty,
                                  regularizationConstant: Double,
                                  learningRate: Double,
                                  learningRateMethod: LBFGSLearningRateMethodTrait,
                                  numberOfExamples: Long) extends DiffFunction[BreezeDenseVector[Double]]{

    private def calculateLoss(
                               data: DataSet[LabeledVector],
                               weightVector: Vector,
                               lossFunction: LBFGSLossFunction)
    : DataSet[Double] = {
      data.map{
        data => lossFunction.loss(data, weightVector)
      }.reduce{
        (left, right) => left + right
      }
    }

    private def calculateGradient(
                                   data: DataSet[LabeledVector],
                                   weightVector: Vector,
                                   lossFunction: LBFGSLossFunction)
    : DataSet[Vector] = {
      data.map {
        data => lossFunction.gradient(data, weightVector)
      }.reduce {
        (left, right) => (left.asBreeze + right.asBreeze).fromBreeze
      }
    }

    override def calculate(weights: BreezeDenseVector[Double]): (Double, BreezeDenseVector[Double]) = {
      val localWeights = weights.fromBreeze
      val num = localWeights.size
      val lossSum = calculateLoss(dataPoints, localWeights, lossFunction).collect().toList.head
      val gradientSum = calculateGradient(dataPoints, localWeights, lossFunction).collect().toList.head
      val lossCount = lossSum / numberOfExamples
      val updatedWeights = regularizationPenalty.takeStep(localWeights, DenseVector.zeros(num).asInstanceOf[Vector], regularizationConstant, learningRateMethod.calculateLearningRate(learningRate, 1, regularizationConstant))
      val loss = regularizationPenalty.regLoss(lossCount, updatedWeights, regularizationConstant)
      val gradientTotal = localWeights.copy
      BLAS.axpy(-1.0, updatedWeights, gradientTotal)
      BLAS.axpy(1.0 / numberOfExamples, gradientSum, gradientTotal)
      (loss, gradientTotal.asBreeze.asInstanceOf[BreezeDenseVector[Double]])
    }
  }

}

/** Implementation of a L-BFGS solver.
  *
  */
object LBFGS {
  def apply() = new LBFGS
}