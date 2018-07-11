# Very Deep Convolutional Networks For Large-Scale Image Recognition
vgg网络研究了卷积网络深度在大规模的图像识别环境下对准确性的影响，在网络中使用的是非常小的 3*3 卷积滤波器.本仓库是对论文中网络的代码实现,使用框架为 Tensorflow,数据格式为 TFRecord.
# Content
1. VGG网络结构图
2. 仓库文件解析
3. 使用本仓库

#### 1.网络结构图
下图描述了论文中研究的 vgg 网络的结构:  
<img src="http://upload-images.jianshu.io/upload_images/3232548-a104f82ae41bc025.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"  height="426" width="495">

#### 2.仓库文件解析
1. 文件夹 /datasets 用于存放数据读取脚本，对应有ImageNet,17flowers等，使用 datasets_factory.py 获取对应的数据读取脚本。  
2. 文件夹 /nets 描述了 vgg 各种结构的网络，如vgg11,vgg16等

#### 3.使用本仓库
