---
layout:     post
title: TensorFlow pb 读写
subtitle: TensorFlow 系列之 pb 文件
date:       2019-5-10
author:     Cong Yu
header-img: img/bg_2.jpg
catalog: true
tags:
    - tensorflow
---


## tensorflow ckpt 转 pb

<details><summary> python 代码 ckpt2pb </summary>
输入 ckpt 路径、保存pb文件路径

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.framework import graph_util
def freeze_graph(model_folder, output_graph="frozen_model.pb"):
    # We retrieve our checkpoint fullpath
    try:
        checkpoint = tf.train.get_checkpoint_state(model_folder)
        input_checkpoint = checkpoint.model_checkpoint_path
        print("[INFO] input_checkpoint:", input_checkpoint)
    except:
        input_checkpoint = model_folder
        print("[INFO] Model folder", model_folder)

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    output_node_names = []  # NOTE: Change here
    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        opts = sess.graph.get_operations()
        for v in opts:
            print(v.name)
            output_node_names.append(v.name)

        # var_list = tf.global_variables()
        # output_node_names_ = [var_list[i].name for i in range(len(var_list))]

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            # input_graph_def,  # The graph_def is used to retrieve the nodes
            sess.graph_def,
            "output/predictions".split(",")  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print("[INFO] output_graph:", output_graph)
        print("[INFO] all done")
```

</details>

## tensorflow 加载 pb 文件

<details><summary> python 代码 ckpt2pb </summary>

**注意**\
但同时加载多个模型是，要 指定图
tf.Graph().as_default() as graph\
sess = tf.Session(graph=graph)
```python
import tensorflow as tf
model_path = ""
with tf.Graph().as_default() as graph:
    with tf.gfile.FastGFile(model_path + 'text_cnn.pb', "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")
    sess = tf.Session(graph=graph)
    x = sess.graph.get_tensor_by_name("placeholder/x:0")
    # is_training = sess.graph.get_tensor_by_name("placeholder/Placeholder:0")  # is_training
    p = sess.graph.get_tensor_by_name("output/logits:0")
```

</details>

