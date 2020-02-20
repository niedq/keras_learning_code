# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:52:01 2019

一个简单的keras神经网络模型

用法，用的时候写在一行
python simple_fc_train.py --dataset animals 
    --model output/simple_nn.model 
    --label-bin output/simple_nn_lb.pickle 
    --plot output/simple_nn_plot.png

@author: ndq
"""




import matplotlib
matplotlib.use("Agg")
#matplotlib.use("Qt5Agg") # 在图形窗口显示


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# 利用argparse定义命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
    help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True, 
    help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True, 
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True, 
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

#初始化数据和标签列表
print("读取图片")
data = []
labels = []

# 利用imutils的paths方法批量读取文件夹下的图片
imagePaths = sorted(list(paths.list_images(args["dataset"])))
#imagePaths = sorted(list(paths.list_images("keras-tutorial/animals")))
random.seed(42)
random.shuffle(imagePaths)

# 利用opencv读取图片
for imagePath in imagePaths:
    img = cv2.imread(imagePath)
    img = cv2.resize(img, (32, 32)).flatten()
    data.append(img)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# 将图片进行标准化，并转换为numpy数组
data = np.array(data, dtype="float") / 255
labels = np.array(labels)

# 将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, 
    labels, test_size=0.25, random_state=42)

# 将标签数据转为数值，并进行独热编码，是标签变为向量
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# 构建简单神经网络，结构为3072-1024-512-3
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# 编译模型
INIT_LR = 0.01
EPOCHS = 75

print("开始训练模型")
opt = SGD(lr=INIT_LR) # 随机梯度下降，设置学习率超参
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 训练模型
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
    batch_size=128, epochs=EPOCHS)

# 训练好的模型进行测试
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))

# 将训练的正确率和损失画图
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# 保存图形和模型
print("保存模型结果")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
print("运行成功")
