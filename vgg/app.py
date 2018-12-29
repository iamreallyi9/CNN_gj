#coding:utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from vgg import vgg16,utils
from vgg.Nclasses import labels

#vgg16是用来计算的，app是用来预测的，Nclass是存储类别参数的，utils是工具--->图像处理工具
#这个project是使用app来运行的，利用已经储存好的参数复原神经网络
#读入图片进行utils.load_image()预处理,img_ready 为[1,224,224,3]
img_path = input('Input the path and image name:')
img_ready = utils.load_image(img_path) 

fig=plt.figure(u"Top-5 预测结果") 

with tf.Session() as sess:
    images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    #使用类进行编程，实例化对象==》使用方法(forward)==》使用属性(prob)
    vgg = vgg16.Vgg16() 
    vgg.forward(images)

    probability = sess.run(vgg.prob, feed_dict={images:img_ready})

    #argsort函数 [argsort=arg+sort] 从小到大排序，从后面取，也就是预测概率最大的五个索引值
    #格式值得进一步研究??probability[0][1000]???


    top5 = np.argsort(probability[0])[-1:-6:-1]
    print ("top5:",top5)
    values = []
    bar_label = []
    #enumerate()遍历后返回索引和内容
    #貌似i是索引
    for n, i in enumerate(top5): 
        print ("n:",n)
        print ("i:",i)
        #把预测值高的五个probility值存入values
        values.append(probability[0][i])
        #把实际标签存入labels
        bar_label.append(labels[i]) 
        print (i, ":", labels[i], "----", utils.percent(probability[0][i]) )

    #以下为绘图表现
    # bar()绘制柱状图，range(len(values)是柱子下标， 是柱子下标，values表示柱高的列，也就是五个预测概率值
    # tick_label是每个柱子上显示的标签（实际对应）widthwidthwidth width是柱 子的宽度， fc 是柱子的颜色
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
    ax.set_ylabel(u'probabilityit') 
    ax.set_title(u'Top-5') 
    for a,b in zip(range(len(values)), values):
        ax.text(a, b+0.0005, utils.percent(b), ha='center', va = 'bottom', fontsize=7)   
    plt.show() 
