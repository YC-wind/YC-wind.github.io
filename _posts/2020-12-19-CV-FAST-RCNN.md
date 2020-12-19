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


![Alt Text]({{ site.baseurl }}/post_images/FAST-RCNN/FAST-RCNN-img3.gif)

## Reference
1. [FAST-RCNN-paper](https://arxiv.org/abs/1504.08083)
2. [FAST-RCNN-github](https://github.com/rbgirshick/fast-rcnn)
3. [ROI Pooling（感兴趣区域池化）](https://blog.csdn.net/H_hei/article/details/89791176)
4. [浅谈深度神经网络 — R-CNN（区域卷积神经网络）](https://zhuanlan.zhihu.com/p/64694855)
5. 