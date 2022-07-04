本项目的测试环境：win10，python3.7，mediapipe0.8.10

各模块介绍：

​		poseembedding.py是人体关键点归一化编码模块

​		poseclassifier.py是人体姿态分类模块，使用的算法是k-NN

​		resultsmooth.py是分类结果平滑模块，使用的是指数移动平均

​		counter.py是运动计数模块

​		visualizer.py是分类结果可视化模块

​		extracttrainingsetkeypoint.py是提取和处理训练集关键点模块，并将特征向量存储在csv文件中

​		trainingsetprocess.py是输入训练集以及训练集的检验校正的模块，里面说明了训练样本文件夹的要求

​		videoprocess.py是检测视频并计数动作的的模块（注意class_name参数的含义）

​		videocapture.py是调用摄像头实时检测并计数动作的模块（注意class_name参数的含义）

​		main.py是整个项目运行的入口程序

​		Roboto-Regular是visualizer.py中需要用到的字体文件

由于我们选择了简单易上手的k-最近邻算法(k-NN) 作为分类器（该算法根据训练集中最接近的样本确定对象的类别），而不是根据各运动的肢体之间的夹角特点作为分类依据，所以该方法具有良好的泛化通用能力，可以广泛应用在诸如深蹲（deep squat）、俯卧撑（push up）、引体向上（pull up）等健身运动的计数上，只需要将训练集更换成对应的运动并修改class_name参数即可。

更详细的说明可参考：https://blog.csdn.net/m0_57110410/article/details/125569971