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

import org.apache.flink.ml.common.{LabeledVector, WeightVector}
import org.apache.flink.ml.math.DenseVector
import org.scalatest.{Matchers, FlatSpec}

import com.github.thssmonkey.LBFGS.RegressionData._
import org.apache.flink.api.scala._

class LBFGSITSuite extends FlatSpec with Matchers with FlinkTestBase {

  // TODO(tvas): Check results again once sampling operators are in place

  behavior of "The LBFGS implementation"
/*
  it should "correctly solve an L1 regularized regression problem" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

    val lbfgs = LBFGS()
      .setStepsize(0.01)
      .setIterations(2000)
      .setLossFunction(lossFunction)
      .setRegularizationPenalty(LBFGSL1Regularization)
      .setRegularizationConstant(0.3)
      .setStorages(10)

    val inputDS: DataSet[LabeledVector] = env.fromCollection(regularizationData)

    val weightVector = lbfgs.optimize(inputDS, None)

    val weights = weightVector.asInstanceOf[DenseVector].data

    expectedRegWeights zip weights foreach {
      case (expectedWeight, weight) =>
        weight should be (expectedWeight +- 0.01)
    }
  }

  it should "correctly perform one step with L2 regularization" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

    val lbfgs = LBFGS()
      .setStepsize(0.1)
      .setIterations(1)
      .setLossFunction(lossFunction)
      .setRegularizationPenalty(LBFGSL2Regularization)
      .setRegularizationConstant(1.0)
      .setStorages(10)

    val inputDS: DataSet[LabeledVector] = env.fromElements(LabeledVector(1.0, DenseVector(3.0)))
    val currentWeights = DenseVector(1.0)

    val weightVector = lbfgs.optimize(inputDS, Some(currentWeights))

    weightVector(0) should be (0.5 +- 0.001)
  }

  it should "estimate a linear function" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

    val lbfgs = LBFGS()
      .setStepsize(1.0)
      .setIterations(800)
      .setLossFunction(lossFunction)
      .setStorages(10)

    val inputDS = env.fromCollection(data)
    val weightVector = lbfgs.optimize(inputDS, None)

    val weights = weightVector.asInstanceOf[DenseVector].data

    expectedWeights zip weights foreach {
      case (expectedWeight, weight) =>
        weight should be (expectedWeight +- 0.1)
    }
  }

  it should "estimate a linear function without an intercept" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

    val lbfgs = LBFGS()
      .setStepsize(0.0001)
      .setIterations(100)
      .setLossFunction(lossFunction)
      .setStorages(10)

    val inputDS = env.fromCollection(noInterceptData)
    val weightVector = lbfgs.optimize(inputDS, None)

    val weights = weightVector.asInstanceOf[DenseVector].data

    expectedNoInterceptWeights zip weights foreach {
      case (expectedWeight, weight) =>
        weight should be (expectedWeight +- 0.1)
    }
  }

  it should "correctly perform one step of the algorithm with initial weights provided" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

    val lbfgs = LBFGS()
      .setStepsize(0.1)
      .setIterations(1)
      .setLossFunction(lossFunction)
      .setStorages(10)

    val inputDS: DataSet[LabeledVector] = env.fromElements(LabeledVector(1.0, DenseVector(3.0)))
    val currentWeights = DenseVector(1.0)

    val weightVector = lbfgs.optimize(inputDS, Some(currentWeights))

    weightVector(0) should be (0.6 +- 0.01)
  }

  it should "terminate early if the convergence criterion is reached" in {
    // TODO(tvas): We need a better way to check the convergence of the weights.
    // Ideally we want to have a Breeze-like system, where the optimizers carry a history and that
    // can tell us whether we have converged and at which iteration

    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

    val lbfgsEarlyTerminate = LBFGS()
      .setConvergenceThreshold(1e2)
      .setStepsize(1.0)
      .setIterations(800)
      .setLossFunction(lossFunction)
      .setStorages(10)

    val inputDS = env.fromCollection(data)

    val weightVectorEarly = lbfgsEarlyTerminate.optimize(inputDS, None)

    val weightsEarly = weightVectorEarly.asInstanceOf[DenseVector].data

    val lbfgsNoConvergence = LBFGS()
      .setStepsize(1.0)
      .setIterations(800)
      .setLossFunction(lossFunction)
      .setStorages(10)

    val weightVectorNoConvergence = lbfgsNoConvergence.optimize(inputDS, None)

    val weightsNoConvergence = weightVectorNoConvergence.asInstanceOf[DenseVector].data

    // Since the first optimizer was set to terminate early, its weights should be different
    weightsEarly zip weightsNoConvergence foreach {
      case (earlyWeight, weightNoConvergence) =>
        weightNoConvergence should not be (earlyWeight +- 0.1)
    }
  }

  it should "come up with similar parameter estimates with xu step-size strategy" in {
    val env = ExecutionEnvironment.getExecutionEnvironment

    env.setParallelism(2)

    val lossFunction = LBFGSGenericLossFunction(LBFGSSquaredLoss, LBFGSLinearPrediction)

    val lbfgs = LBFGS()
      .setStepsize(1.0)
      .setIterations(800)
      .setLossFunction(lossFunction)
      .setLearningRateMethod(LBFGSLearningRateMethod.Xu(-0.75))
      .setStorages(10)

    val inputDS = env.fromCollection(data)
    val weightVector = lbfgs.optimize(inputDS, None)

    val weights = weightVector.asInstanceOf[DenseVector].data

    expectedWeights zip weights foreach {
      case (expectedWeight, weight) =>
        weight should be (expectedWeight +- 0.1)
    }
  }
  // TODO: Need more corner cases, see sklearn tests for SGD linear model
*/
}
