**train-test**中代码为训练与评估综合脚本，其中CONTINUE-EPOCH参数为每次运行训练次数，每次训练完成会自动进行一次预测评估
如果仅要评估则将该参数设为0即可，该参数在代码开头十分好找，如果需要重新训练则需删除results里面相关文件夹
训练与预测前应该先将相关数据集放在data中



尽管已经去掉将近一半的参数对比等重要性较低代码，但本实验所需代码仍然较多，其文件相关部分命名解释如下

1，2，3，4对应任务1，2，3，4代码部分，似乎CUDA以及一写版本原因在我们构建的小样本电影数据集上效果有一定差异

我们在本地电脑和云GPU上存在这个问题，但是结果均支持我的结论

ALL为FER和RAF联合数据集
FER，RAF为两个数据集
CBAM，Swin窗口，Swinmove窗口移动，Transformer四个注意力机制，其中加了注意力机制的代码较多，其实没必要一一测试，大体都一样，但是1-ALL.py的模型作为后续迁移训练的源模型必须训练
movie为封面人脸训练
review为评论文本训练
blend为模态融合训练
blance为样本均衡
move为迁移训练
mid，late为中期，晚期融合
attention为注意力机制融合
Emotic为连续情感数据集

挑战任务中camera1和camera-adjusted分别是加入滤波窗口前后的摄像头情感识别，自监督代码为视频分析

需要将想要分析的视频放在data的vidio文件夹里面，每次分析一个视频，要分析不同视频需要在代码里面该文件名



**data-utils**里面为数据爬取，其它相关人脸截取代码，因为评论的清洗用的是在云GPU上下载的模型，以及文件架构的变化，
云GPU与实际电脑上运行架构区别，以及后续评论和封面人脸的合并等，该部分的移植确实存在难度，最重要的是需要人工标注，
而且也不是项目重点，所以未进行展示，数据处理上的代码建议仅仅作为思路展示，不建议进行复现的操作，数据集建议直接获取使用



**model**里面为BERT模型



云GPU上具体环境配置
一、核心基础环境
Python：3.12.3（Miniconda 打包）
解释器路径：/root/miniconda3/bin/python
环境类型：miniconda base 基础环境，无独立虚拟 env
系统：Ubuntu Linux 5.15 64 位 x86\_64
二、CUDA \& GPU（深度学习核心）
PyTorch：2.7.0+cu128
CUDA runtime 版本：12.8
显卡：单张 RTX 4080 SUPER，GPU 可用状态正常（还有用过其它型号显卡，如4090，5090）
系统虚拟包标注\_\_cuda=13.0是系统驱动最高支持版本，不影响 pytorch cu128 运行
三、包安装路径（导入优先级）
所有第三方库统一安装在：
/root/miniconda3/lib/python3.12/site-packages
无额外自定义代码搜索目录，导入库只会读取 base 的 site-packages。
四、PATH 路径优先级
最高优先级是 miniconda base 的 bin 目录，执行python/pip/conda都会调用 base 内工具，不会调用系统自带 python。
包含 CUDA 二进制目录 /usr/local/cuda/bin，编译 cuda 算子正常。



依赖包

一、Python 内置标准库

os

re

random

collections（Counter）

二、数值 / 数据处理库

numpy

pandas

三、图像 / 视觉处理库

PIL（Pillow，内含 Image、ImageFont、ImageDraw）

cv2（OpenCV）

mtcnn

四、绘图可视化库

matplotlib（plt、FigureCanvasAgg）

seaborn

五、机器学习评估工具（sklearn）

sklearn.metrics（confusion\_matrix）

sklearn.preprocessing（LabelEncoder）

六、深度学习框架 PyTorch 全套

torch

torch.nn

torch.optim

torch.nn.functional (F)

torch.utils.data（DataLoader、Dataset、TensorDataset）

torchvision（datasets、transforms、models、ResNet18\_Weights）

七、进度条工具

tqdm

八、NLP 大模型库

transformers（BertTokenizer、BertModel）

一键安装所有库指令

pip install torch torchvision pillow opencv-python mtcnn numpy pandas matplotlib seaborn scikit-learn tqdm transformers

