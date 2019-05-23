---
layout:     post
title: （转）机器学习-LogisticRegression
subtitle: 机器学习系列之 LogisticRegression
date:       2019-5-23
author:     Cong Yu
header-img: img/bg_2.jpg
catalog: true
tags:
    - 机器学习
---

## Logistic 回归 概述

`Logistic 回归 或者叫逻辑回归 虽然名字有回归，但是它是用来做分类的。其主要思想是: 根据现有数据对分类边界线(Decision Boundary)建立回归公式，以此进行分类。`

## 须知概念

### Sigmoid 函数

#### 回归 概念

假设现在有一些数据点，我们用一条直线对这些点进行拟合（这条直线称为最佳拟合直线），这个拟合的过程就叫做回归。进而可以得到对这些点的拟合直线方程，那么我们根据这个回归方程，怎么进行分类呢？请看下面。

#### 二值型输出分类函数

我们想要的函数应该是: 能接受所有的输入然后预测出类别。例如，在两个类的情况下，上述函数输出 0 或 1.或许你之前接触过具有这种性质的函数，该函数称为 `海维塞得阶跃函数(Heaviside step function)`，或者直接称为 `单位阶跃函数`。然而，海维塞得阶跃函数的问题在于: 该函数在跳跃点上从 0 瞬间跳跃到 1，这个瞬间跳跃过程有时很难处理。幸好，另一个函数也有类似的性质（可以输出 0 或者 1 的性质），且数学上更易处理，这就是 Sigmoid 函数。 Sigmoid 函数具体的计算公式如下: 

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

下图给出了 Sigmoid 函数在不同坐标尺度下的两条曲线图。当 x 为 0 时，Sigmoid 函数值为 0.5 。随着 x 的增大，对应的 Sigmoid 值将逼近于 1 ; 而随着 x 的减小， Sigmoid 值将逼近于 0 。如果横坐标刻度足够大， Sigmoid 函数看起来很像一个阶跃函数。

因此，为了实现 Logistic 回归分类器，我们可以在每个特征上都乘以一个回归系数（如下公式所示），然后把所有结果值相加，将这个总和代入 Sigmoid 函数中，进而得到一个范围在 0~1 之间的数值。任何大于 0.5 的数据被分入 1 类，小于 0.5 即被归入 0 类。所以，Logistic 回归也是一种概率估计，比如这里Sigmoid 函数得出的值为0.5，可以理解为给定数据和参数，数据被分入 1 类的概率为0.5。想对Sigmoid 函数有更多了解，可以点开[此链接](https://www.desmos.com/calculator/bgontvxotm)跟此函数互动。

### 基于最优化方法的回归系数确定

Sigmoid 函数的输入记为 z ，由下面公式得到: 

$$z = f(\mathbf{x}) = w_1x_1 + w_2x_2 + ... + w_nx_n + b, b=w_0x_0$$

如果采用向量的写法，上述公式可以写成 $$z = W^{T}X$$ ，它表示将这两个数值向量对应元素相乘然后全部加起来即得到 z 值。其中的向量 x 是分类器的输入数据，向量 w 也就是我们要找到的最佳参数（系数），从而使得分类器尽可能地精确。为了寻找该最佳参数，需要用到最优化理论的一些知识。我们这里使用的是——梯度上升法（Gradient Ascent）。


## Logistic 回归 原理

### Logistic 回归 工作原理

```
每个回归系数初始化为 1
重复 R 次:
    计算整个数据集的梯度
    使用 步长 x 梯度 更新回归系数的向量
返回回归系数
```

### Logistic 回归 开发流程

```
收集数据: 采用任意方法收集数据
准备数据: 由于需要进行距离计算，因此要求数据类型为数值型。另外，结构化数据格式则最佳。
分析数据: 采用任意方法对数据进行分析。
训练算法: 大部分时间将用于训练，训练的目的是为了找到最佳的分类回归系数。
测试算法: 一旦训练步骤完成，分类将会很快。
使用算法: 首先，我们需要输入一些数据，并将其转换成对应的结构化数值；接着，基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于哪个类别；在这之后，我们就可以在输出的类别上做一些其他分析工作。
```

### Logistic 回归 算法特点

```
优点: 计算代价不高，易于理解和实现。
缺点: 容易欠拟合，分类精度可能不高。
适用数据类型: 数值型和标称型数据。
```
code：
```python
'''
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
'''
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #convert to NumPy matrix
    labelMat = mat(classLabels).transpose() #convert to NumPy matrix
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):              #heavy on matrix operations
        h = sigmoid(dataMatrix*weights)     #matrix mult
        error = (labelMat - h)              #vector subtraction
        weights = weights + alpha * dataMatrix.transpose()* error #matrix mult
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not 
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))
        

```

#### 最小二乘法

回归任务最常用的性能度量是**均方误差（mean squared error, MSE）**。首先介绍**单变量线性回归**，试想我们要在二维平面上拟合一条曲线，则每个样例（即每个点）只包含一个实值属性（x值）和一个实值输出标记（y值），此时均方误差可定义为：

$$E(f;D) = \frac{1}{m} \sum_{i=1}^m(y_i-f(x_i))^2\\
\qquad\qquad = \frac{1}{m} \sum_{i=1}^m(y_i-wx_i-b)^2$$

有时我们会把这样描述模型总误差的式子称为**损失函数**或者**目标函数**（当该式是优化目标的时候）。这个函数的自变量是模型的参数 $$w$$ 和 $$b$$。由于给定训练集时，样本数 $$m$$ 是一个确定值，也即常数，所以可以把 $$\frac{1}{m}$$ 这一项拿走。

**最小二乘法（least square method）**就是基于均方误差最小化来进行模型求解的一种方法，寻找可使损失函数值最小的参数 $$w$$ 和 $$b$$ 的过程称为最小二乘**参数估计（parameter estimation）**。

通过对损失函数分别求参数 $$w$$ 和 $$b$$ 的偏导，并且令导数为0，可以得到这两个参数的**闭式（closed-form）解**（也即**解析解**）：

$$w = \frac{\sum_{i=1}^m y_i(x_i - \bar{x})}{\sum_{i=1}^m x_i^2 - \frac{1}{m}(\sum_{i=1}^m x_i)^2}$$

$$b = \frac{1}{m} \sum_{i=1}^m (y_i-wx_i)$$

在实际任务中，只要我们把自变量（x, y, m）的值代入就可以求出数值解了。

为什么可以这样求解呢？因为损失函数是一个**凸函数**（记住是向下凸，类似U型曲线），导数为0表示该函数曲线最低的一点，此时对应的参数值就是能使均方误差最小的参数值。特别地，**要判断一个函数是否凸函数，可以求其二阶导数**，若二阶导数在区间上非负则称其为凸函数，若在区间上恒大于零则称其为**严格凸函数**。

#### 多元线性回归

前面是直线拟合，样例只有一个属性。对于样例包含多个属性的情况，我们就要用到**多元线性回归（multivariate linear regression）**（又称作多变量线性回归）了。

令 $$\mathbf{\hat{w}} = (\mathbf{w};b)$$。把数据集表示为 $$m \times (d+1)$$ 大小的矩阵，每一行对应一个样例，前 $$d$$ 列是样例的 $$d$$ 个属性，**最后一列恒置为1**，对应偏置项。把样例的实值标记也写作向量形式，记作 $$\mathbf{y}$$。则此时损失函数为：

$$E_{\mathbf{\hat{w}}} = (\mathbf{y} - X\mathbf{\hat{w}})^T (\mathbf{y} - X\mathbf{\hat{w}})$$

同样使用最小二乘法进行参数估计，首先对 $$\mathbf{\hat{w}}$$ 求导：

$$\frac{\partial E_{\mathbf{\hat{w}}}}{\partial \mathbf{\hat{w}}} = 2 X^T(X\mathbf{\hat{w}} - \mathbf{y})$$

令该式值为0可得到 $$\mathbf{\hat{w}}$$ 的闭式解：

$$\mathbf{\hat{w}}* = (X^TX)^{-1}X^T\mathbf{y}$$

这就要求 $$X^TX$$ 必须是可逆矩阵，也即必须是**满秩矩阵（full-rank matrix）**，这是线性代数方面的知识，书中并未展开讨论。但是！**现实任务中 $$X^TX$$ 往往不是满秩的**，很多时候 $$X$$ 的列数很多，甚至超出行数（例如推荐系统，商品数是远远超出用户数的），此时 $$X^TX$$ 显然不满秩，会解出多个 $$\mathbf{\hat{w}}$$。这些解都能使得均方误差最小化，这时就需要由学习算法的**归纳偏好**决定了，常见的做法是引入**正则化（regularization）**项。

#### 广义线性模型

除了直接让模型预测值逼近实值标记 $$y$$，我们还可以让它逼近 $$y$$ 的衍生物，这就是**广义线性模型（generalized linear model）**的思想，也即：

$$y = g^{-1}(\mathbf{w^Tx} + b)$$

其中 $$g(\cdot)$$ 称为**联系函数（link function）**，要求单调可微。使用广义线性模型我们可以实现强大的**非线性函数映射**功能。比方说**对数线性回归（log-linear regression）**，令 $$g(\cdot) = ln(\cdot)$$，此时模型预测值对应的是**实值标记在指数尺度上的变化**：

$$\ln y = \mathbf{w^Tx} + b$$

## 对数几率回归（逻辑回归）

前面说的是线性模型在回归学习方面的应用，这节开始就是讨论分类学习了。

线性模型的输出是一个实值，而分类任务的标记是离散值，怎么把这两者联系起来呢？其实广义线性模型已经给了我们答案，我们要做的就是**找到一个单调可微的联系函数**，把两者联系起来。

对于一个二分类任务，比较理想的联系函数是**单位阶跃函数（unit-step function）**：

$$y =
\begin{cases}
0& \text{z<0;}\\
0.5& \text{z=0;}\\
1& \text{z>0,}
\end{cases}$$

但是单位阶跃函数不连续，所以不能直接用作联系函数。这时思路转换为**如何近似单位阶跃函数**呢？**对数几率函数（logistic function）**正是我们所需要的（注意这里的 $$y$$ 依然是实值）：

$$y = \frac{1}{1+e^{-z}}$$

对数几率函数有时也称为对率函数，是一种**Sigmoid函数**（即形似S的函数）。将它作为 $$g^-(\cdot)$$ 代入广义线性模型可得：

$$y = \frac{1}{1+ e^{-(\mathbf{w^Tx} + b)}}$$

该式可以改写为：

$$\ln{\frac{y}{1-y}} = \mathbf{w^Tx} + b$$

其中，$$\frac{y}{1-y}$$ 称作**几率（odds）**，我们可以把 $$y$$ 理解为该样本是正例的概率，把 $$1-y$$ 理解为该样本是反例的概率，而几率表示的就是**该样本作为正例的相对可能性**。若几率大于1，则表明该样本更可能是正例。对几率取对数就得到**对数几率（log odds，也称为logit）**。几率大于1时，对数几率是正数。

由此可以看出，对数几率回归的实质使用线性回归模型的预测值逼近分类任务真实标记的对数几率。它有几个优点：

1. 直接对分类的概率建模，无需实现假设数据分布，从而避免了假设分布不准确带来的问题；
2. 不仅可预测出类别，还能得到该预测的概率，这对一些利用概率辅助决策的任务很有用；
3. 对数几率函数是任意阶可导的凸函数，有许多数值优化算法都可以求出最优解。

#### 最大似然估计

有了预测函数之后，我们需要关心的就是怎样求取模型参数了。这里介绍一种与最小二乘法异曲同工的办法，叫做**极大似然法（maximum likelihood method）**。我在另一个项目中有这方面比较详细的讲解，欢迎前往[项目主页](https://github.com/familyld/SYSU_Data_Mining/tree/master/Linear_Regression)交流学习。

前面说道可以把 $$y$$ 理解为一个样本是正例的概率，把 $$1-y$$ 理解为一个样本是反例的概率。而所谓极大似然，就是最大化预测事件发生的概率，也即**最大化所有样本的预测概率之积**。令 `$$p(c=1|x)$$` 和 `$$p(c=0|x)$$` 分别代表 $$y$$ 和 $$1-y$$。（注：书中写的是 $$y=1$$ 和 $$y=0$$，这里为了和前面的 $$y$$ 区别开来，我用了 $$c$$ 来代表标记）。简单变换一下公式，可以得到：

$$p(c=1|\mathbf{x}) = \frac{e^(\mathbf{w^Tx} + b)}{1+e^{\mathbf{w^Tx} + b}}$$

$$p(c=0|\mathbf{x}) = \frac{1}{1+e^{\mathbf{w^Tx} + b}}$$

但是！由于预测概率都是小于1的，如果直接对所有样本的预测概率求积，所得的数会非常非常小，当样例数较多时，会超出精度限制。所以，一般来说会对概率去对数，得到**对数似然（log-likelihood）**，此时**求所有样本的预测概率之积就变成了求所有样本的对数似然之和**。对率回归模型的目标就是最大化对数似然，对应的似然函数是：

$$\ell(\mathbf{w},b) = \sum_{i=1}^m \ln p(c_i | \mathbf{x_i;w};b)\\
= \sum_{i=1}^m \ln (c_ip_1(\hat{\mathbf{x_i}};\beta) + (1-c_i)p_0(\hat{\mathbf{x_i}};\beta))$$

可以理解为若标记为正例，则加上预测为正例的概率，否则加上预测为反例的概率。其中 $$\beta = (\mathbf{w};b)$$。

对该式求导，令导数为0可以求出参数的最优解。特别地，我们会发现似然函数的导数和损失函数是等价的，所以说**最大似然解等价于最小二乘解**。最大化似然函数等价于最小化损失函数：

$$E(\beta) = \sum_{i=1}^m (-y_i\beta^T\hat{x_i} + \ln (1+e^{\beta^T\mathbf{\hat{x_i}}}))$$

这是一个关于 $$\beta$$ 的高阶可导连续凸函数，可以用最小二乘求（要求矩阵的逆，计算开销较大），也可以用数值优化算法如**梯度下降法（gradient descent method）**、**牛顿法（Newton method）**等逐步迭代来求最优解（可能陷入局部最优解）。

## 线性判别分析（LDA）

#### 二分类

**线性判别分析（Linear Discriminant Analysis，简称LDA）**，同样是利用线性模型，LDA提供一种不同的思路。在LDA中，我们不再是拟合数据分布的曲线，而是**将所有的数据点投影到一条直线上**，使得**同类点的投影尽可能近，不同类点的投影尽可能远**。二分类LDA最早有Fisher提出，因此也称为**Fisher判别分析**。

具体来说，投影值 $$y = \mathbf{w}^T\mathbf{x}$$，我们不再用 $$y$$ 逼近样例的真实标记，而是希望同类样例的投影值尽可能相近，异类样例的投影值尽可能远离。如何实现呢？首先，同类样例的投影值尽可能相近意味着**同类样例投影值的协方差应尽可能小**；然后，异类样例的投影值尽可能远离意味着**异类样例投影值的中心应尽可能大**。合起来，就等价于最大化：

$$J = \frac{\Vert \mathbf{w}^T\mu_0 - \mathbf{w}^T\mu_1 \Vert^2_2}{\mathbf{w}^T\Sigma_0\mathbf{w}+\mathbf{w}^T\Sigma_1\mathbf{w}}\\
= \frac{\mathbf{w}^T(\mu_0 - \mu_1)(\mu_0 - \mu_1)^T\mathbf{w}}{\mathbf{w}^T(\Sigma_0+\Sigma_1)\mathbf{w}}$$

其中，分子的 $$\mu_i$$ 表示第i类样例的**均值向量**（即表示为向量形式后对各维求均值所得的向量）。分子表示的是两类样例的均值向量投影点（也即类中心）之差的 $$\ell_2$$ 范数的平方，这个值越大越好。 分母中的 $$\Sigma_i$$ 表示第i类样例的**协方差矩阵**。分母表示两类样例投影后的协方差之和，这个值越小越好。

定义**类内散度矩阵（within-class scatter matrix）**：

$$S_w = \sigma_0 + \sigma_1\\
= \sum_{x \in X_0} (\mathbf{x} - \mu_0)(\mathbf{x} - \mu_0)^T + \sum_{x \in X_1} (\mathbf{x} - \mu_1)(\mathbf{x} - \mu_1)^T$$

定义**类间散度矩阵（between-class scatter matrix）**：

$$S_b = (\mu_0 - \mu_1)(\mu_0 - \mu_1)^T$$

这两个矩阵的规模都是 $$d\times d$$，其中 $$d$$ 是样例的维度（属性数目）。于是可以重写目标函数为：

$$J = \frac{\mathbf{w}^T S_b \mathbf{w}}{\mathbf{w}^T S_w \mathbf{w}}$$

也即 $$S_b$$ 和 $$S_w$$ 的**广义瑞利熵（generalized Rayleigh quotient）**。

可以注意到，分子和分母中 $$w$$ 都是二次项，因此，**最优解与 $$w$$ 的大小无关，只与方向有关**。

令分母为1，用拉格朗日乘子法把约束转换为方程，再稍加变换我们便可以得出：

$$\mathbf{w} = S_w^{-1}(\mu_0 - \mu_1)$$

但一般不直接对矩阵 $$S_w$$ 求逆，而是采用**奇异值分解**的方式。

#### 多分类

多分类LDA与二分类不同在于，学习的是一个规模为 $$d \times d'$$ 的投影矩阵 $$\mathbf{W}$$，而不是规模为 $$d \times 1$$ 的投影向量 $$\mathbf{w}$$。这个投影矩阵把样本投影到 $$d'$$ 维空间（或者说 $$d'$$ 维超平面）上，由于 $$d'$$ 通常远小于样例原来的属性数目 $$d$$，且投影过程用到了类别信息（标记值），所以LDA也常常被视为一种**监督降维技术**。（注：$$d'$$ 最大可取为类别数-1）

## 多分类学习

有些二分类学习方法（如LDA）可以直接推广到多分类，但现实中我们更多地是**基于一些策略，把多分类任务分解为多个二分类任务**，利用二分类模型来解决问题。有三种最经典的拆分策略，分别是一对一，一对其余，和多对多。

#### 一对一

**一对一（One vs. One，简称OvO）**的意思是把所有类别两两配对。假设样例有N个类别，OvO会产生 $$\frac{N(N-1)}{2}$$ 个子任务，**每个子任务只使用两个类别的样例**，并产生一个对应的二分类模型。测试时，新样本输入到这些模型，产生 $$\frac{N(N-1)}{2}$$ 个分类结果，最终预测的标记由投票产生，也即把被预测得最多的类别作为该样本的类别。

#### 一对其余

**一对其余（One vs. Rest，简称OvR）**在有的文献中也称为**一对所有（One vs. All，简称OvA）**，但这种说法并不严谨。因为OvR产生 $$N$$ 个子任务，**每个任务都使用完整数据集**，把一个类的样例当作正例，其他类的样例当作反例，所以应该是一对其余而非一对所有。OvR产生 $$N$$ 个二分类模型，测试时，新样本输入到这些模型，产生 $$N$$ 个分类结果，若只有一个模型预测为正例，则对应的类别就是该样本的类别；若有多个模型预测为正例，则选择置信度最大的类别（参考模型评估与选择中的**比较检验**）。

OvO和OvR各有优劣：OvO需要训练的分类器较多，因此**OvO的存储开销和测试时间开销通常比OvR更大**；OvR训练时要用到所有样例，因此**OvR的训练时间开销通常比OvO更大**。测试性能取决于具体的数据分布，大多情况下这两个拆分策略都差不多。

#### 多对多

**多对多（Many vs. Many，简称MvM）**是每次将多个类作为正例，其他的多个类作为反例。OvO和OvR都是MvM的特例。书中介绍的是一种比较常用的MvM技术——**纠错输出码（Error Correcting Outputs Codes，简称ECOC）**。

MvM的正反例划分不是任意的，必须有特殊的构造，否则组合起来时可能就无法定位到预测为哪一类了。ECOC的工作过程分两步：

- 编码：对应于训练。假设有N个类别，计划做M次划分，每次划分把一部分类别划为正类，一部分类别划分为反类，最终训练出M个模型。而每个类别在M次划分中，被划为正类则记作+1，被划为负类则记作-1，于是可以表示为一个M维的编码。

- 解码：对应于预测。把新样本输入M个模型，所得的M个预测结果组成一个预测编码。把这个预测编码和各个类别的编码进行比较，跟哪个类别的编码距离最近就预测为哪个类别。

类别划分由**编码矩阵（coding matrix）**指定，编码矩阵有多重形式，常见的有二元码（正类为+1，负类为-1）和三元码（多出了**停用类**，用0表示，因为有停用类的存在，**训练时可以不必使用全部类别的样例**）。举个三元码的例子：

||f1<br>$$\downarrow$$|f2<br>$$\downarrow$$|f3<br>$$\downarrow$$|f4<br>$$\downarrow$$|f5<br>$$\downarrow$$|海明距离<br>$$\downarrow$$|欧氏距离<br>$$\downarrow$$|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|$$C_1\rightarrow$$| -1 | +1 | -1 | +1 | +1 | 4 | 4 |
|$$C_2\rightarrow$$| +1 | -1 | -1 | +1 | -1 | 2 | 2 |
|$$C_3\rightarrow$$| -1 | +1 | +1 | -1 | +1 | 5 | 2$$\sqrt{5}$$  |
|$$C_4\rightarrow$$| -1 | -1 | +1 | +1 | -1 | 3 | $$\sqrt{10}$$ |
|测试样本$$\rightarrow$$| -1 | -1 | +1 | -1 | +1 | - | - |

这里一共有4个类别，对应每一行。计划做5次划分，得到f1至f5共五个二分类器，对应每一列。可以看到每一个类别有一个5位的编码表示。测试时，把新样本输入到5个模型，得到预测编码。然后计算这个预测编码和每个类别编码的距离。这里举了海明距离（不同的位数的数目）和欧氏距离作为例子。可以看到测试样本与类别2的距离最近，因此预测该样本为类别2。

特别地，为什么称这种方法为**纠错**输出码呢？因为**ECOC编码对分类器的错误有一定的容忍和修正能力**。即使预测时某个分类器预测成了错误的编码，在解码时仍然有机会产生正确的最终结果。具体来说，对同一个学习任务，**编码越长，纠错能力越强**。但是相应地也需要训练更多分类器，增大了计算和存储的开销。

对同等长度的编码来说，理论上**任意两个类别之间的编码距离越远，纠错能力越强**。但实际任务中我们一般不需要获取最优编码，一方面非最优编码已经能产生不错的效果；另一方面，**即使获得理论上最优的编码，实际性能也不一定最好**。因为机器学习还涉及其他一些方面，在划分多个类时产生的新问题难度往往也不同，有可能理论最优的编码产生的类别子集难以区分，问题难度更大，从而使得性能下降。

## 类别不平衡问题

**类别不平衡（class-imbalance）**问题非常普遍，比方说推荐系统中用户购买的商品（通常视作正例）和用户未购买的商品（通常视作反例）比例是极为悬殊的。如果直接用类别不平衡问题很严重的数据集进行训练，所得模型会严重偏向所占比例较大的类别。**本节默认正类样例较少，负类样例较多**。这里主要介绍三种做法：

#### 欠采样

**欠采样（undersampling）**针对的是负类，也即移取训练集的部分反例，使得正类和负类的样例数目相当。由于丢掉了大量反例，所以**时间开销也大大减少**。但是带来一个问题就是，**随机丢弃反例可能会丢失一些重要信息**。书中提到一种解决方法是利用**集成学习机制**，将反例划分为多个集合，用于训练不同的模型，从而使得**对每个模型来说都进行了欠采样，但全局上并无丢失重要信息**。

#### 过采样

**过采样（oversampling）**针对的是正类，也即增加训练集的正例，使得正类和负类的样例数目相当。过采样的**时间开销会增大很多**，因为需要引入很多正例。注意！过采样**不能简单地通过重复正例来增加正例的比例**，这样会引起严重的过拟合问题。一种较为常见的做法是对已有正例进行**插值**来产生新的正例。

#### 阈值移动

**阈值移动（threshold-moving）**利用的是**再缩放**思想。回想前面对数几率回归中，几率 $$\frac{y}{1-y}$$ 表示正例的相对可能性，我们默认以1作为阈值，其实是假设了样本的真实分布为正例反例各一半。但这可能不是真相，假设我们有一个存在类别不平衡问题的训练集，正例数目为 $$m^+$$，反例数目为 $$m^-$$，可以重定义：

$$\frac{y'}{1-y'} = \frac{y}{1-y} \times \frac{m^-}{m^+}$$

这就是**再缩放（rescaling）**。当几率大于 $$\frac{m^+}{m^-}$$ 时就预测为正例。但必须注意，这种思想是**基于观测几率近似真实几率这一假设**的，现实任务中这一点未必成立。

* * *

## Reference

* **参考1：[familyld/Machine_Learning](https://github.com/familyld/Machine_Learning)**
* **作者：[片刻](http://cwiki.apachecn.org/display/~jiangzhonglian) [1988](http://cwiki.apachecn.org/display/~lihuisong)**
* **地址：[GitHub地址](https://github.com/apachecn/AiLearning)**
* **版权声明：信息来源于 [ApacheCN](http://www.apachecn.org/)**
