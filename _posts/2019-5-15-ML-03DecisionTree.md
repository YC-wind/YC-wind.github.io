---
layout:     post
title: （转）机器学习-DecisionTree
subtitle: 机器学习系列之 DecisionTree
date:       2019-5-15
author:     Cong Yu
header-img: img/bg_2.jpg
catalog: true
tags:
    - 机器学习
---


## 决策树 概述

`决策树（Decision Tree）算法是一种基本的分类与回归方法，是最经常使用的数据挖掘算法之一。我们这章节只讨论用于分类的决策树。`

`决策树模型呈树形结构，在分类问题中，表示基于特征对实例进行分类的过程。它可以认为是 if-then 规则的集合，也可以认为是定义在特征空间与类空间上的条件概率分布。`

`决策树学习通常包括 3 个步骤：特征选择、决策树的生成和决策树的修剪。`

## 决策树 场景

一个叫做 "二十个问题" 的游戏，游戏的规则很简单：参与游戏的一方在脑海中想某个事物，其他参与者向他提问，只允许提 20 个问题，问题的答案也只能用对或错回答。问问题的人通过推断分解，逐步缩小待猜测事物的范围，最后得到游戏的答案。

决策树的定义：

分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点（node）和有向边（directed edge）组成。结点有两种类型：内部结点（internal node）和叶结点（leaf node）。内部结点表示一个特征或属性(features)，叶结点表示一个类(labels)。

用决策树对需要测试的实例进行分类：从根节点开始，对实例的某一特征进行测试，根据测试结果，将实例分配到其子结点；这时，每一个子结点对应着该特征的一个取值。如此递归地对实例进行测试并分配，直至达到叶结点。最后将实例分配到叶结点的类中。

## 决策树 原理

信息熵(`information-entropy`)是度量样本集合纯度最常用的一种指标。信息增益大的特征具有更强的分类能力。假设$$X$$是一个取有限值的离散随机变量，其概率分布为 
$$P(X=x_i) = p_i, i=1,2,3…,n$$
, 则随机变量 $$X$$ 的熵定义为.

​ $$H(X)=-\sum_{i=1}^{n}{p_i}{log(p_i)}$$

条件熵(`conditional-entropy`)设有随机变量 $$(X,Y)$$ ,其联合概率分布为：

$$P(X=x_i, Y=y_i)=p_{ij}, i = 1,2…,n; j=1,2,…,m$$

。条件熵

$$H(Y\|X)$$

表示在已知随机变量$$X$$的条件下随机变量$$Y$$的不确定性。

​$$H(Y\|X)=\sum_{i=1}^{n}{P(X=x_i)H(Y\|X=x_i)}$$

当熵和条件熵的概率由数据估计(极大似然估计)得到时，所对应的熵与条件熵分别称为经验熵和经验条件熵。

信息增益(`information-gain`) 表示得知特征$$X$$的信息而使得类$$Y$$的信息的不确定度减少的程度。

也就是熵的减少或者是数据无序度的减少。

特征$$A$$对训练数据集$$D$$的信息增益$$g(D,A)$$, 定义为集合$D$的经验熵$$H(D)$$与特征$$A$$给定条件下$$D$$的经验条件熵

$$H(D\|A)$$

之差，即

​$$g(D, A)=H(D) - H(D\|A)$$

⚠️ 一般来说，熵$$H(X)$$与条件熵

$$H(Y\|X)$$

之差称为互信息，决策树中信息增益等价于训练数据集中类与特征的互信息。

信息增益(率)比

特征$A$ 对训练数据集$D$的信息增益比 定义为信息增益$$g(D,A)$$ 与训练数据集$$D$$关于特征$$A$$的值的熵$$H_A(D)$$之比。即如下所示：

​ $$g_R(D,A)=\frac{g(D, A)}{H_A(D)}$$

其中，

$$H_A(D)=-\sum_{i}^{n}\frac{\|D_i\|}{\|D\|}*log_2(\frac{\|D_i\|}{{\|D\|}})$$

 基尼指数
 
基尼指数(`Gini- index`) 反映了从数据集$D$中随机抽取两个样本，其类别标记不一致的概率。数据集$D$的纯度可以用基尼指数值来度量：$$Gini(D)$$越小，则数据集$$D$$的纯度越高。

​ $$Gini(D) = \sum_{k=1}^{\|y\|}\sum_{k^{'}\neq{k}{p}}{p_k}{p_{k^{'}}}$$

​ $$=1-\sum_{k}^{\|y\|}{p_{k}^{2}}$$

属性$a$ 的基尼指数定义为：

​ $$GiniIndex(D,a) = \sum_{v=1}^{V}\frac{\|D^v\|}{\|D\|}*Gini(D^v)$$

于是我们在候选属性集合A中，选择那个使得划分后基尼指数最小的属性作为最优划分属性，即

​ $$a_*= arg_{a\in A}min Gini-index(D,a)$$


基尼值是另一种衡量样本集纯度的指标。反映的是从一个数据集中随机抽取两个样本，其类别标志不同的概率。

基尼值越小，样本集的纯度越高。

由基尼值引伸开来的就是基尼指数这种准则了，基尼指数越小，表示使用属性 $$a$$ 划分后纯度的提升越大。

### 决策树 开发流程

```
收集数据：可以使用任何方法。
准备数据：树构造算法 (这里使用的是ID3算法，只适用于标称型数据，这就是为什么数值型数据必须离散化。 还有其他的树构造算法，比如CART)
分析数据：可以使用任何方法，构造树完成之后，我们应该检查图形是否符合预期。
训练算法：构造树的数据结构。
测试算法：使用训练好的树计算错误率。
使用算法：此步骤可以适用于任何监督学习任务，而使用决策树可以更好地理解数据的内在含义。
```

### 决策树 算法特点

```
优点：计算复杂度不高，输出结果易于理解，数据有缺失也能跑，可以处理不相关特征。
缺点：容易过拟合。
适用数据类型：数值型和标称型。
```

### 决策树 python 代码

```python
import numpy as np

class DecisionTree:
    """决策树使用方法：
    
        - 生成实例： clf = DecisionTrees(). 参数mode可选，ID3或C4.5，默认C4.5
        
        - 训练，调用fit方法： clf.fit(X,y).  X,y均为np.ndarray类型
                            
        - 预测，调用predict方法： clf.predict(X). X为np.ndarray类型
    
    """
    def __init__(self,mode='C4.5'):
        self._tree = None
        
        if mode == 'C4.5' or mode == 'ID3':
            self._mode = mode
        else:
            raise Exception('mode should be C4.5 or ID3')
        
            
    
    def _calcEntropy(self,y):
        """
        函数功能：计算熵
        参数y：数据集的标签
        """
        num = y.shape[0]
        #统计y中不同label值的个数，并用字典labelCounts存储
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys(): labelCounts[label] = 0
            labelCounts[label] += 1
        #计算熵
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/num
            entropy -= prob * np.log2(prob)
        return entropy
    
    
    
    def _splitDataSet(self,X,y,index,value):
        """
        函数功能：返回数据集中特征下标为index，特征值等于value的子数据集
        """
        ret = []
        featVec = X[:,index]
        X = X[:,[i for i in range(X.shape[1]) if i!=index]]
        for i in range(len(featVec)):
            if featVec[i]==value:
                ret.append(i)
        return X[ret,:],y[ret]
    
    
    def _chooseBestFeatureToSplit_ID3(self,X,y):
        """ID3
        函数功能：对输入的数据集，选择最佳分割特征
        参数dataSet：数据集，最后一列为label
        主要变量说明：
                numFeatures：特征个数
                oldEntropy：原始数据集的熵
                newEntropy：按某个特征分割数据集后的熵
                infoGain：信息增益
                bestInfoGain：记录最大的信息增益
                bestFeatureIndex：信息增益最大时，所选择的分割特征的下标
        """
        numFeatures = X.shape[1]
        oldEntropy = self._calcEntropy(y)
        bestInfoGain = 0.0
        bestFeatureIndex = -1
        #对每个特征都计算一下infoGain，并用bestInfoGain记录最大的那个
        for i in range(numFeatures):        
            featList = X[:,i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            #对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵，
            #进一步地可以计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X,sub_y = self._splitDataSet(X,y,i,value)
                prob = len(sub_y)/float(len(y))
                newEntropy += prob * self._calcEntropy(sub_y)  
            #计算信息增益，根据信息增益选择最佳分割特征
            infoGain = oldEntropy - newEntropy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeatureIndex = i
        return bestFeatureIndex
        
    def _chooseBestFeatureToSplit_C45(self,X,y):
        """C4.5
            ID3算法计算的是信息增益，C4.5算法计算的是信息增益比，对上面ID3版本的函数稍作修改即可
        """
        numFeatures = X.shape[1]
        oldEntropy = self._calcEntropy(y)
        bestGainRatio = 0.0
        bestFeatureIndex = -1
        #对每个特征都计算一下gainRatio=infoGain/splitInformation
        for i in range(numFeatures):        
            featList = X[:,i]
            uniqueVals = set(featList)
            newEntropy = 0.0
            splitInformation = 0.0
            #对第i个特征的各个value，得到各个子数据集，计算各个子数据集的熵，
            #进一步地可以计算得到根据第i个特征分割原始数据集后的熵newEntropy
            for value in uniqueVals:
                sub_X,sub_y = self._splitDataSet(X,y,i,value)
                prob = len(sub_y)/float(len(y))
                newEntropy += prob * self._calcEntropy(sub_y)  
                splitInformation -= prob * np.log2(prob)
            #计算信息增益比，根据信息增益比选择最佳分割特征
            #splitInformation若为0，说明该特征的所有值都是相同的，显然不能作为分割特征
            if splitInformation==0.0:
                pass
            else:
                infoGain = oldEntropy - newEntropy
                gainRatio = infoGain/splitInformation
                if(gainRatio > bestGainRatio):
                    bestGainRatio = gainRatio
                    bestFeatureIndex = i
        return bestFeatureIndex
    
    
    
    def _majorityCnt(self,labelList):
        """
        函数功能：返回labelList中出现次数最多的label
        """
        labelCount={}
        for vote in labelList:
            if vote not in labelCount.keys(): labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.iteritems(),key=lambda x:x[1], reverse=True)
        return sortedClassCount[0][0]
    
    
    
    def _createTree(self,X,y,featureIndex):
        """建立决策树
        featureIndex，类型是元组，它记录了X中的特征在原始数据中对应的下标。
        """
        labelList = list(y)
        #所有label都相同的话，则停止分割，返回该label
        if labelList.count(labelList[0]) == len(labelList): 
            return labelList[0]
        #没有特征可分割时，停止分割，返回出现次数最多的label
        if len(featureIndex) == 0:
            return self._majorityCnt(labelList)
        
        #可以继续分割的话，确定最佳分割特征
        if self._mode == 'C4.5':
            bestFeatIndex = self._chooseBestFeatureToSplit_C45(X,y)
        elif self._mode == 'ID3':
            bestFeatIndex = self._chooseBestFeatureToSplit_ID3(X,y)
            
        bestFeatStr = featureIndex[bestFeatIndex]
        featureIndex = list(featureIndex)
        featureIndex.remove(bestFeatStr)
        featureIndex = tuple(featureIndex)
        #用字典存储决策树。最佳分割特征作为key，而对应的键值仍然是一棵树（仍然用字典存储）
        myTree = {bestFeatStr:{}}
        featValues = X[:,bestFeatIndex]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            #对每个value递归地创建树
            sub_X,sub_y = self._splitDataSet(X,y, bestFeatIndex, value)
            myTree[bestFeatStr][value] = self._createTree(sub_X,sub_y,featureIndex)
        return myTree  
    
    def fit(self,X,y):
        #类型检查
        if isinstance(X,np.ndarray) and isinstance(y,np.ndarray):
            pass
        else: 
            try:
                X = np.array(X)
                y = np.array(y)
            except:
                raise TypeError("numpy.ndarray required for X,y")
        
        featureIndex = tuple(['x'+str(i) for i in range(X.shape[1])])
        self._tree = self._createTree(X,y,featureIndex)
        return self  #allow chaining: clf.fit().predict()

    

    def predict(self,X):
        if self._tree==None:
            raise NotFittedError("Estimator not fitted, call `fit` first")
        
        #类型检查
        if isinstance(X,np.ndarray): 
            pass
        else: 
            try:
                X = np.array(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        
        def _classify(tree,sample):
            """
            用训练好的决策树对输入数据分类 
            决策树的构建是一个递归的过程，用决策树分类也是一个递归的过程
            _classify()一次只能对一个样本（sample）分类
            To Do: 多个sample的预测怎样并行化？
            """
            featIndex = tree.keys()[0]
            secondDict = tree[featIndex]
            key = sample[int(featIndex[1:])]
            valueOfkey = secondDict[key]
            if isinstance(valueOfkey, dict): 
                label = _classify(valueOfkey,sample)
            else: label = valueOfkey
            return label
            
        if len(X.shape)==1:
            return _classify(self._tree,X)
        else:   
            results = []
            for i in range(X.shape[0]):
                results.append(_classify(self._tree,X[i]))
            return np.array(results)
        
    def show(self):
        if self._tree==None:
            raise NotFittedError("Estimator not fitted, call `fit` first")
        
        #plot the tree using matplotlib
        import treePlotter
        treePlotter.createPlot(self._tree)

     
class NotFittedError(Exception):
    """
    Exception class to raise if estimator is used before fitting
    
    """
    pass
```

* * *

## Reference

* **参考1：[familyld/Machine_Learning](https://github.com/familyld/Machine_Learning)**
* **作者：[片刻](http://cwiki.apachecn.org/display/~jiangzhonglian) [1988](http://cwiki.apachecn.org/display/~lihuisong)**
* **地址：[GitHub地址](https://github.com/apachecn/AiLearning)**
* **版权声明：信息来源于 [ApacheCN](http://www.apachecn.org/)**
