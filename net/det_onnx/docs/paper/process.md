# 毕业论文流程
## 第一步：搭建环境

1.将电脑重装成ubutnu20.04

2.安装nvidia driver

3.安装cuda 

4.安装cudnn

5.安装Miniconda

6.创建虚拟环境

7.安装pytorch等python库

## 第二步：搭建项目
1.下载项目代码

2.安装依赖包

3.了解注意力机制

4.在yolov11中加入注意力机制


## 第三步：模型训练
1.训练加入了CBAM注意力的yolov11模型

2.微调以注意力为核心的yolov12模型

3.将模型导出成onnx格式

4.对模型进行量化

## 第四步：numpy重构
1.理解yolo的模型输出

2.了解nms的原理

3.了解map的计算方法

4.使用numpy，onnx进行重构推理以及验证代码

## 第五步：模型部署
1.在远程服务器上写出界面

2.将模型部署到远程服务器上

3.在远程服务器上测试模型


## 第六步：后续工作
1.对模型采取剪枝，替换backbone，知识蒸馏等方法优化模型

2.对远程服务器上的模型推理进行并发

3.对远程服务器添加实时检测功能