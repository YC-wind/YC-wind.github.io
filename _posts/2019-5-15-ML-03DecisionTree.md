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

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

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

熵（entropy）：
熵指的是体系的混乱的程度，在不同的学科中也有引申出的更为具体的定义，是各领域十分重要的参量。

\\(Ent(D) = -\sum_{k=1}^{|\mathcal{Y}|}p_klog_2p_k\\)

其中 \\(|\mathcal{Y}|$ 为类别集合，\\(p_k$ 为该类样本占样本总数的比例。

信息论（information theory）中的熵（香农熵）：
是一种信息的度量方式，表示信息的混乱程度，也就是说：信息越有序，信息熵越低。例如：火柴有序放在火柴盒里，熵值很低，相反，熵值很高。

信息增益（information gain）：
在划分数据集前后信息发生的变化称为信息增益。

\\(Gain(D,a) = Ent(D) - \sum_{v=1}^{V}\frac{|D^v|}{|D|}Ent(D^v)\\)

增益率（gain ratio）是C4.5算法采用的选择准则，定义如下：

\\(Gain_ratio(D,a) = \frac{Gain(D,a)}{IV(a)}\\)

其中，

\\(IV(a) = -\sum_{v=1}^V\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}\\)

基尼指数
基尼指数（Gini index）是CART算法采用的选择准则，定义如下：

基尼值：

\\(Gini(D) = \sum_{k=1}^{|\mathcal{Y}|}\sum_{k' \neq k}p_kp_{k'}\ =1-\sum_{k=1}^{|\mathcal{Y}|}p_k^2\\)

基尼指数：

\\(Gini_index(D,a) = \sum_{v=1}^{V}\frac{|D^v|}{|D|}Gini(D^v)\\)

基尼值是另一种衡量样本集纯度的指标。反映的是从一个数据集中随机抽取两个样本，其类别标志不同的概率。

基尼值越小，样本集的纯度越高。

由基尼值引伸开来的就是基尼指数这种准则了，基尼指数越小，表示使用属性 \\(a\\) 划分后纯度的提升越大。

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


* * *

## Reference

* **参考1：[familyld/Machine_Learning](https://github.com/familyld/Machine_Learning)**
* **作者：[片刻](http://cwiki.apachecn.org/display/~jiangzhonglian) [1988](http://cwiki.apachecn.org/display/~lihuisong)**
* **地址：[GitHub地址](https://github.com/apachecn/AiLearning)**
* **版权声明：信息来源于 [ApacheCN](http://www.apachecn.org/)**
