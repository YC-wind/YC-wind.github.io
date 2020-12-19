---
layout:     post
title: FASTer-RCNN
subtitle: CV-DETECTION 系列之 FASTer-RCNN
date:       2020-12-19
author:     Cong Yu
header-img: img/bg_2.jpg
catalog: true
tags:
    - FASTer-RCNN
    - DETECTION
---
## FASTer-RCNN

经过R-CNN和Fast RCNN的积淀，Ross B. Girshick在2016年提出了新的Faster RCNN，在结构上，Faster RCNN已经将特征抽取(feature extraction)，proposal提取，bounding box regression(rect refine)，classification都整合在了一个网络中，使得综合性能有较大提高，在检测速度方面尤为明显。

![img.png]({{ site.baseurl }}/post_images/FASTer-RCNN/FASTer-RCNN-img0.png)

### 主要架构

Faster RCNN其实可以分为4个主要内容：

- **Conv layers**。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。

- _**Region Proposal Networks**_。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。

![img.png]({{ site.baseurl }}/post_images/FASTer-RCNN/FASTer-RCNN-img0.png)

- **Roi Pooling**。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。

- **Classification**。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。



## Reference
1. [FASTer-RCNN-paper](https://zhuanlan.zhihu.com/p/31426458)
2. [FASTer-RCNN-github](https://github.com/ShaoqingRen/faster_rcnn)
3. [一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)
4. [捋一捋pytorch官方FasterRCNN代码](https://zhuanlan.zhihu.com/p/145842317)
5. [图像检测算法](https://www.cnblogs.com/cyssmile/p/13573811.html)