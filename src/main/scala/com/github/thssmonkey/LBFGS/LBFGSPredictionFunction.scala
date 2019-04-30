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

import org.apache.flink.ml.math.{BLAS, Vector}

/** An abstract class for prediction functions to be used in optimization **/
abstract class LBFGSPredictionFunction extends Serializable {
  def predict(features: Vector, weights: Vector): Double

  def gradient(features: Vector, weights: Vector): Vector
}

/** A linear prediction function **/
object LinearPrediction extends LBFGSPredictionFunction {
  override def predict(features: Vector, weightVector: Vector): Double = {
    BLAS.dot(features, weightVector)
  }

  override def gradient(features: Vector, weights: Vector): Vector = {
    features.copy
  }
}