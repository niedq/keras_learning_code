# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:11:11 2020

调用之前定义的卷积神经网络，进行训练，并保存相关结果

整体流程：定义各类传入参数 => 读取图片文件 => resize图片大小，标准化图片并转为numpy数组
         => 图片切分训练集测试集，标签独热编码 => 初始化图片增强，训练要用fit_generator
         => 调用定义好的神经网络框架类 => 确定相关超参，编译模型 => 训练模型 
         => 查看模型在测试集效果，画图展示模型效果 => 保存相关结果
用法，写在一行
python simple_vgg_train.py --dataset animals 
    --model output/simplevggnet.model
    --label-bin output/simplevggnet_lb.pickle
    --plot output/simplevggnet_plot.png


@author: ndq
"""

from model.simple_model import SimpleModel
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# 定义命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, 
    help='path to input dataset of images')
ap.add_argument('-m', '--model', required=True,
    help='path to output trained model')
ap.add_argument('-l', '--label-bin', required=True,
    help='path to output label binarizer')
ap.add_argument('-p', '--plot', required=True, 
    help='path to output accuracy/loss plot')
args = vars(ap.parse_args())

# 初始化数据和标签
print('开始读取数据')
data = []
labels = []

# 读取数据并打乱顺序
#imagePaths = sorted(list(paths.list_images("animals")))
imagePaths = sorted(list(paths.list_images(args['dataset'])))
random.seed(33)
random.shuffle(imagePaths)

# 利用opencv读取图片
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    
# 数据转为numpy数组，并标准化
data = np.array(data, dtype='float') / 255
labels = np.array(labels)

# 数据拆分为训练集合测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, 
    test_size=0.25, random_state=33)

# 对标签进行独热编码
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# 图片增强
aug = ImageDataGenerator(rotation_range=30, #旋转范围
    width_shift_range=0.1, #水平平移范围
    height_shift_range=0.1, #垂直平移范围
    shear_range=0.2, #透视变换的范围
    zoom_range=0.2, #缩放范围
    horizontal_flip=True, #水平反转
    fill_mode='nearest' #填充模式
    )

# 调用定义好的神经网络框架，输入参数
model = SimpleModel.build(width=64, height=64, channels=3, classes=len(lb.classes_))

# 确定学习率，epoch，batch_size超参
learnrate = 0.01
epochs = 75
bs = 32

# 模型编译
print('开始训练模型')
opt = SGD(lr=learnrate, decay=learnrate / epochs) #decay参数控制学习率衰减
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# 模型训练
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=bs), 
    validation_data=(X_test, y_test), steps_per_epoch=len(X_train) // bs, 
    epochs=epochs)

# 验证模型
print('模型验证')
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), 
    predictions.argmax(axis=1), target_names=lb.classes_))

# 画图展示模型效果
N = np.arange(0, epochs)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history['loss'], label='train_loss')
plt.plot(N, H.history['val_loss'], label='val_loss')
plt.plot(N, H.history['acc'], label='train_acc')
plt.plot(N, H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy (SampleModel)')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
#plt.savefig("output/smallvggnet_plot.png")
plt.savefig(args['plot'])

# 保存相关结果
print('保存神经网络相关结果')
#model.save("output/samplevggnet.model")
model.save(args['model'])
#f = open("output/samplevggnet_lb.pickle", "wb")
f = open(args['label_bin'], 'wb')
f.write(pickle.dumps(lb))
f.close()
print('运行成功')

