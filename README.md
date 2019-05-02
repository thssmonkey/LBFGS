# **L-BFGS - FlinkML算法**

基于Flink的L-BFGS算法实现

## **介绍**

- L-BFGS全称是Limited-memory BFGS，是拟牛顿算法族的优化算法，使用有限量的计算机存储来近似Broyden-Fletcher-Goldfarb-Shanno（BFGS）算法

- 是一种无约束最优化算法，是解无约束非线性规划问题常用的方法，具有收敛速度快、内存开销少等优点

- 算法的目标是在实变量的无约束值$$x$$上最小化$$f(x)$$

## **背景**

金融数据欺诈检测是一个持久性的研究话题，涉及到金融业，政府，企业部门和普通消费者等各方各面； 传统方法依赖于诸如审计之类的手工技术，不仅耗时，昂贵且不准确。因此，研究学者和组织机构已经转向使用统计和计算方法的自动化流程，比如基于数据挖掘的欺诈检测已经得到了长足的发展。

由于Flink是一个针对流数据和批数据的分布式处理引擎，以数据并行和流水线方式执行任意流数据程序，具有吞吐量大，延迟低，支持状态处理等优势。而且支持机器学习算法（FlinkML库）。因此在Flink上进行基于数据挖掘的金融欺诈检测是一个值得研究的问题。这其中就涉及到机器学习算法的应用。

然而FlinkML中只实现了少部分算法，其中优化框架下的**L-BFGS**算法还未实现。

因此我实现了这个算法，您通过引用相关包即可在Flink中直接使用L-BFGS算法。

## **使用**

### 引入：

使用Flink大多通过maven来进行管理（我的环境配置是Flink 1.7.1 + Scala 2.11.12 + Maven 3.3.9），可以通过maven方式导入包，在`pom.xml`中的`<dependencies> </dependencies>`中引入下面配置：

```xml
<dependency>
  <groupId>com.github.thssmonkey</groupId>
  <artifactId>LBFGS</artifactId>
  <version>1.0.3</version>
</dependency>
```

其它引入方式：

[引入方式链接]: https://search.maven.org/artifact/com.github.thssmonkey/LBFGS/1.0.3/jar

### 使用：

在代码中import相关LBFGS的API：

```scala
import com.github.thssmonkey.LBFGS._
```

### 版本：

|     版本      | 状态 |
| :-----------: | :--: |
|     1.0.3     | 稳定 |
| 1.0.1 ~ 1.0.2 | 无效 |

## **Demo**

```scala
import org.apache.flink.api.scala._
import org.apache.flink.ml.common._
import org.apache.flink.ml.math._

/**
  * 1. import LBFGS相关API
  */
import com.github.thssmonkey.LBFGS._

/**
  * Test Flink's LBFGS for a simple linear regression use case using the advertisement dataset from ISL
  * @see https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv
  *
  * args(0) should be the local path of the dataset.
  */
object Demo extends App{
  val env = ExecutionEnvironment.getExecutionEnvironment

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
  val lossFunction = LBFGSGenericLossFunction(SquaredLoss, LBFGSLinearPrediction)
  
  /**
    * 2. 创建LBFGS实例，并设置参数
    */
  val lbfgs = LBFGS()
    .setLossFunction(lossFunction)
    .setIterations(1000)
    //.setStepsize(0.0001)
    .setConvergenceThreshold(0.001)
    .setStorages(10)

  val initialWeights = Some(DenseVector.zeros(3).asInstanceOf[Vector])
  
  /**
    * 3. 运行得到结果
    */
  val weights = lbfgs.optimize(training, initialWeights)
  println(weights)

  test.map { l => (l, weights) }
   	  .map(x => (x._1.vector.dot(x._2), x._1.label, x._2))
  	  .map(x =>(SquaredLoss.loss(x._1, x._2) / 50, x._3))
      .sum(0)
      .print()
}


/**
  * result:
  * DenseVector(0.05488324053036031, 0.2156819005580397, 0.016836151594152405)
  * (1.8643101949719494,DenseVector(0.05488324053036031, 0.2156819005580397, 0.016836151594152405))
  */
```

## **API**



**API文档下载链接**：

[LBFGS-API-LocalSite]: https://cloud.tsinghua.edu.cn/d/ac69926eec824605bbde/

## **开源**

**地址**：

[mvnrepository仓库]: https://mvnrepository.com/artifact/com.github.thssmonkey/LBFGS

或

[中央仓库]: https://search.maven.org/search?q=g:com.github.thssmonkey



