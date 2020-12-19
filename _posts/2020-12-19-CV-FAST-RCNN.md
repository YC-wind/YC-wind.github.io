---
layout:     post
title: FAST-RCNN
subtitle: CV-DETECTION 系列之 FAST-RCNN
date:       2020-12-19
author:     Cong Yu
header-img: img/bg_2.jpg
catalog: true
tags:
    - FAST-RCNN
    - DETECTION
---
## FAST-RCNN

### 任务介绍

```
目标检测（Object Detection) 将目标的分割和识别合二为一，通俗点说就是给定一张图片要精确的定位到物体所在位置，并完成对物体类别的识别。
```



RCNN检测慢主要体现为：

![图]({{ site.baseurl }}/post_images/FAST-RCNN/FAST-RCNN-img1.png)

相比于RCNN，主要改进：
- RCNN大缺点：由于每一个候选框都要独自经过CNN，耗时多。通过共享卷积层，现在不是每一个候选框都当做输入进入CNN了，最后一个卷积层后加了一个ROI pooling layer。
- 实现end-to-end（端对端）单阶段训练，使用多任务损失函数。
- 所有层都可以fine-tune
- 不需要离线存储特征文件

![图]({{ site.baseurl }}/post_images/FAST-RCNN/FAST-RCNN-img2.png)

### 主要框架

![图]({{ site.baseurl }}/post_images/FAST-RCNN/FAST-RCNN-img.png)


R-CNN框架图对比，可以发现主要有两处不同：一是最后一个卷积层后加了一个ROI pooling layer，二是损失函数使用了多任务损失函数(multi-task loss)，将边框回归Bounding Box Regression直接加入到CNN网络中训练。

(1) ROI pooling layer实际上是SPP-NET的一个精简版，SPP-NET对每个proposal使用了不同大小的金字塔映射，而ROI pooling layer只需要下采样到一个7x7的特征图。对于VGG16网络conv5_3有512个特征图，这样所有region proposal对应了一个7*7*512维度的特征向量作为全连接层的输入。

spp-net图结构：
![图]({{ site.baseurl }}/post_images/FAST-RCNN/FAST-RCNN-img4.png)
ROI pooling：
![Alt Text]({{ site.baseurl }}/post_images/FAST-RCNN/FAST-RCNN-img3.gif)

换言之，这个网络层可以把不同大小的输入映射到一个固定尺度的特征向量，而我们知道，conv、pooling、relu等操作都不需要固定size的输入，因此，在原始图片上执行这些操作后，虽然输入图片size不同导致得到的feature map尺寸也不同，不能直接接到一个全连接层进行分类，但是可以加入这个神奇的ROI Pooling层，对每个region都提取一个固定维度的特征表示，再通过正常的softmax进行类型识别。

(2) R-CNN训练过程分为了三个阶段，而Fast R-CNN直接使用softmax替代SVM分类，同时利用多任务损失函数边框回归也加入到了网络中，这样整个的训练过程是端到端的(除去Region Proposal提取阶段)。

也就是说，之前R-CNN的处理流程是先提proposal，然后CNN提取特征，之后用SVM分类器，最后再做bbox regression，而在Fast R-CNN中，作者巧妙的把bbox regression放进了神经网络内部，与region分类和并成为了一个multi-task模型，实际实验也证明，这两个任务能够共享卷积特征，并相互促进。


所以，Fast-RCNN很重要的一个贡献是成功的让人们看到了Region Proposal + CNN这一框架实时检测的希望，原来多类检测真的可以在保证准确率的同时提升处理速度，也为后来的Faster R-CNN做下了铺垫。

## Reference
1. [FAST-RCNN-paper](https://arxiv.org/abs/1504.08083)
2. [FAST-RCNN-github](https://github.com/rbgirshick/fast-rcnn)
3. [ROI Pooling（感兴趣区域池化）](https://blog.csdn.net/H_hei/article/details/89791176)
4. [浅谈深度神经网络 — R-CNN（区域卷积神经网络）](https://zhuanlan.zhihu.com/p/64694855)
5. [SPP-net原理解读](https://www.cnblogs.com/chaofn/p/9305374.html)