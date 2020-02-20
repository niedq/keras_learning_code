# keras_learning_code
使用Keras的深度学习训练神经网络并用于自己的数据，完成一个简单示例
示例目的：构建简单全连接神经网络和小型vgg神经网络进行图片分类（猫，狗，熊猫）

# simple_fc_train.py用法
python simple_fc_train.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png

# simple_vgg_train.py用法
python simple_vgg_train.py --dataset animals --model output/simplevggnet.model --label-bin output/simplevggnet_lb.pickle --plot output/simplevggnet_plot.png

# predict.py用法
python predict.py --image images/dog.jpg --model output/simplevggnet.model --label-bin output/simplevggnet_lb.pickle --width 64 --height 64 --flatten -1 
