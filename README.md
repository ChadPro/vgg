# Very Deep Convolutional Networks For Large-Scale Image Recognition
vgg网络研究了卷积网络深度在大规模的图像识别环境下对准确性的影响，在网络中使用的是非常小的 3*3 卷积滤波器.本仓库是对论文中网络的代码实现,使用框架为 Tensorflow,数据格式为 TFRecord.
# Content
1. VGG网络结构图
2. 仓库文件解析
3. 制作训练数据集
4. 使用本仓库

#### 1.网络结构图
下图描述了论文中研究的 vgg 网络的结构:  
<img src="http://upload-images.jianshu.io/upload_images/3232548-a104f82ae41bc025.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"  height="399" width="495">

#### 2.仓库文件解析
1. 文件夹 /datasets 用于存放数据读取脚本，对应有ImageNet,17flowers等，使用 datasets_factory.py 获取对应的数据读取脚本。  
2. 文件夹 /nets 描述了 vgg 各种结构的网络，如vgg11,vgg16等  
3. /preprocessing 数据预处理脚本  
4. /model 模型文件存放位置  
5. /board_log 用于tensorflow board的log文件存放位置  
6. /back_up 临时备份使用  

#### 3.训练数据集
1. 17flowers:  
[data下载地址](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)  

2. Cat&Dog  
[data下载地址](https://www.kaggle.com/c/dogs-vs-cats)  

3. ImageNet:  
[ImageNet下载地址]()  

#### 4.使用本仓库
1. train for 17flowers    
使用脚本 vgg_train.py  
dataset:对应数据集的数据脚本
train_data_path:是train data路径
val_data_path:是val data路径
num_classes:是分类数目
```
python vgg_train.py --dataset='flowers17_224' --train_data_path='train.tfrecord' --val_data_path='val.tfrecord' --num_classes=17
```

2. train for cat&dog  

3. train for ImageNet  

4. fine tune  
Now, if we have a trained model, we can use this model to fine tune a new model, to classify new things.  

5. Detection for one image  

6. Detection with camera  


