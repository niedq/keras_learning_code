# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 09:49:36 2020

利用构建好的模型预测新的图片

用法, 写在一行
python predict.py --image images/dog.jpg
    --model output/simplevggnet.model
    --label-bin output/simplevggnet_lb.pickle
    --width 64 
    --height 64 
    --flatten -1 

@author: ndq
"""

from keras.models import load_model
import argparse
import pickle
import cv2

# 利用argparse定义命令行参数
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
    help='path to input image we are going to classify')
ap.add_argument('-m', '--model', required=True,
    help='path to trained Keras model')
ap.add_argument('-l', '--label-bin', required=True,
    help='path to label binarizer')
ap.add_argument('-w', '--width', type=int, default=64, 
    help='target spatial dimension width')
ap.add_argument('-e', "--height", type=int, default=64, 
    help='target spatial dimension height')
ap.add_argument('-f', '--flatten', type=int, default=-1, 
    help='whether or not we should flatten the image')
args = vars(ap.parse_args())

# 读取图片并且将图片resize
image = cv2.imread(args['image'])
#image = cv2.imread("images/dog.jpg")
output = image.copy()
image = cv2.resize(image, (args['width'], args['height']))
#image = cv2.resize(image, (64, 64))

# 将图片数据标准化到[0,1]
image = image.astype('float') / 255

# 根据命令行参数，判断是否对图片进行展开
if args['flatten'] > 0:
    image = image.flatten()
    image = image.reshape((1, image.shape[0]))
else:
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# 读取训练好的模型和二值化标签
print('读取训练好的模型和标签')
model = load_model(args['model'])
#model = load_model("output/samplevggnet.model")
lb = pickle.loads(open(args['label_bin'], 'rb').read())
#lb = pickle.loads(open("output/samplevggnet_lb.pickle", "rb").read())

# 对图片进行预测
pred = model.predict(image)

# 输出图片最大可能性的类别
i = pred.argmax(axis=1)[0]
label = lb.classes_[i]

# 将图片类别和概率画在输出的图像上
text = '{}: {:.2f}%'.format(label, pred[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
    (0, 0, 255), 2)

# 展示图片
print("展示图片")
cv2.imshow('Image', output)
cv2.waitKey(0)



