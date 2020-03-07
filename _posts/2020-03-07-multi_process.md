---
layout:     post
title: python多进程-multiprocessing
subtitle: 多进程-介绍
date:       2020-03-07
author:     Cong Yu
header-img: img/bg_2.jpg
catalog: true
tags:
    - 多进程
---

## python 多进程 process

`Process 类用来描述一个进程对象。创建子进程的时候，只需要传入一个执行函数和函数的参数即可完成 Process 示例的创建`

multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)

target 是函数名字，需要调用的函数 \
args 函数需要的参数，以 tuple 的形式传入 \

star() 方法启动进程 \
join() 方法实现进程间的同步，等待所有进程退出。\
close() 用来阻止多余的进程涌入进程池 Pool 造成进程阻塞。\
is_alive() 判断进程状态。\
terminate() 杀死进程。 \

### multiprocessing 代码示例

```python
import multiprocessing
import os
 
def run_proc(name):
    print('Child process {0} {1} Running '.format(name, os.getpid()))
 
if __name__ == '__main__':
    print('Parent process {0} is Running'.format(os.getpid()))
    for i in range(5):
        p = multiprocessing.Process(target=run_proc, args=(str(i),))
        print('process start')
        p.start()
    p.join()
    print('Process close')
```

运行结果 \

```
Parent process 13478 is Running
process start
process start
Child process 0 13479 Running 
process start
Child process 1 13480 Running 
process start
process start
Child process 2 13481 Running 
Child process 3 13482 Running 
Child process 4 13483 Running 
Process close
```

## **进程管理模型**

模型预测进程需要使用到 Pipe ，用来在两个进程间通信，两个进程分别位于管道的两端。

```python
class PredictServer:

    def __init__(self):
        self.version = {}

    def func(self, conn):  # conn管道类型
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        import tensorflow as tf
        has_model = False
        try:
            from predict_model import predict_batch
            a = predict_batch(["xxx"], ["xxx"])
            has_model = True
            print(a)
        except:
            print("model not found...")
        while True:
            temp = conn.recv()
            flag_ = True
            try:
                json_data = json.loads(temp)
                if has_model:
                    res = predict_batch(json_data["text_as"], json_data["text_bs"])
                else:
                    res = "not found model"
            except Exception as e:
                print(repr(e))
                res = "sorry, error"
            # print(res)
            conn.send(res)  # 发送的数据
            # flag = flag_ and self.robot_version.get(key, None) is not None and self.robot_version[key][3]
            print(f"子进程：{os.getpid()} ，接受数据：{temp}，返回：{res}")

    def predict_batch(self, text_as, text_bs):
        json_data = {
            "text_as": text_as,
            "text_bs": text_bs
        }
        key = "relation_model"
        json_data_string = json.dumps(json_data)
        # 检测，所有死掉的进程，全部下线
        pops = []
        for k, v in self.version.items():
            if not v[2].is_alive():
                pops.append(k)
        [self.version.pop(_) for _ in pops]
        if self.version.get(key, None):
            print("has model")
            [conn_a, conn_b, p, _] = self.version[key]
            if p.is_alive():
                conn_b.send(json_data_string)
                a = conn_b.recv()
            else:
                a = "model is close"
            return a
        else:
            print(f"init model {key}")
            conn_a, conn_b = multiprocessing.Pipe()  # 创建一个管道，两个口
            print("ok1")
            p = multiprocessing.Process(target=self.func, args=(conn_a,))
            p.daemon = True
            self.version[key] = [conn_a, conn_b, p, True]
            p.start()
            self.version[key] = [conn_a, conn_b, p, True]
            print("ok2")
            conn_b.send(json_data_string)
            print("ok3")
            a = conn_b.recv()
            print("ok4")
            return a
            
p_model = PredictServer()
p_model.predict_batch(["xxx"],["xxx"])
```

模型管理类回自动检测是否创建进程，之后就会一直使用 后台预测模型服务，进行预测。

### 多版本切换功能


```python
class PredictServer:

    def __init__(self):
        self.version = {}

    def func(self, conn, version):  # conn管道类型
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        import tensorflow as tf
        has_model = False
        key = f"{version}"
        try:
            bsm = Model(version=version)
            a = bsm.predict("xxx")
            has_model = True
            print(a)
        except:
            print("model not found...")
        while True:
            temp = conn.recv()
            flag_ = True
            try:
                json_data = json.loads(temp)
                if json_data["version"] == version:
                    if has_model:
                        res = bsm.predict(json_data["text"])
                    else:
                        res = "not found model"
                else:
                    # 目前设置更新模型时，把之前的模型关闭掉
                    res = "sorry, not match"
            except Exception as e:
                print(repr(e))
                res = "sorry, error"
            # print(res)
            conn.send(res)  # 发送的数据
            flag = flag_ and self.version.get(key, None) is not None and self.version[key][3]
            print(f"进程状态：{flag}，子进程：{os.getpid()} ，接受数据：{temp}，返回：{res}")

    def predict(self, text, version):
        json_data = {
            "version": version,
            "text": text
        }
        json_data_string = json.dumps(json_data)
        key = f"{version}"
        # 检测，所有死掉的进程，全部下线
        pops = []
        for k, v in self.version.items():
            if not v[2].is_alive():
                pops.append(k)
        [self.version.pop(_) for _ in pops]
        if self.version.get(key, None):
            print("has model")
            [conn_a, conn_b, p, _] = self.version[key]
            if p.is_alive():
                conn_b.send(json_data_string)
                a = conn_b.recv()
            else:
                a = "model is close"
            return a
        else:
            print(f"init model {version}")
            conn_a, conn_b = multiprocessing.Pipe()  # 创建一个管道，两个口
            print("ok1")
            p = multiprocessing.Process(target=self.func, args=(conn_a, version))
            p.daemon = True
            self.version[key] = [conn_a, conn_b, p, True]
            p.start()
            self.version[key] = [conn_a, conn_b, p, True]

            # 其他版本 下线
            pops = []
            for k, v in self.version.items():
                if key != k:
                    v[2].terminate()
                    print("stop process")
                    v[2].join()
                    pops.append(k)

            [self.version.pop(_) for _ in pops]
            print("ok2")
            conn_b.send(json_data_string)
            print("ok3")
            a = conn_b.recv()
            print("ok4")
            return a
            
p_model = PredictServer()
version = 1
a = p_model.predict("xxx", version)
```

模型管理类回自动检测是否创建进程，同时自动切换版本（其他版本默认失效，当然修改策略保存也可）。

### 模型进程启动与下线

模型启动
```python
version = 1
a = p_model.predict("xxx", version)
```

模型下线
```python
for k,v in p_model.version.items():
    print(k,v)
    if v[2].is_alive():
        v[2].terminate()
        print("stop process")
        v[2].join() 
```

模型状态
```python
for k,v in p_model.version.items():
    print(k, v[2].is_alive())
```

* * *

## Reference

* **参考1：[Python3多进程multiprocess学习](https://blog.csdn.net/qhd1994/article/details/79864087)**
