import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV2_SIZE = 5
NUM_CONV1_KERNELS = 32
NUM_CONV2_KERNELS = 64
FC_SIZE = 512
OUT_NODE = 10
def forward(x,train,regularizer):
    #例x=[100,28,28,1],conv_w1=[5,5,1,32],con_b1=[32]
    #relu1=[100,28,28,32],pool1=[100,14,14,32]
    #con_w2=[5,5,32,64].con_b2=[64]
    #relu2=[100,14,14,64],pool2=[100,7,7,64]
    #pool2_lenth=[3136],reshaped=[100,7*7*64]

    con_w1 = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,NUM_CONV1_KERNELS],regularizer)
    con_b1 = get_bias([NUM_CONV1_KERNELS])
    relu1 = tf.nn.relu(tf.nn.bias_add(conv2d(x,con_w1),con_b1))
    pool1 = max_pool2x2(relu1)

    con_w2 = get_weight([CONV2_SIZE,CONV2_SIZE,NUM_CONV1_KERNELS,NUM_CONV2_KERNELS],regularizer)
    con_b2 = get_bias([NUM_CONV2_KERNELS])
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2d(pool1,con_w2),con_b2))
    pool2 = max_pool2x2(relu2)

    pool2_shape = pool2.get_shape().as_list()
    pool2_lenth = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    reshaped = tf.reshape(pool2,[pool2_shape[0],pool2_lenth])

    w1 = get_weight([pool2_lenth,FC_SIZE],regularizer)
    b1 = get_bias(FC_SIZE)
    y1 = tf.nn.relu(tf.matmul(reshaped,w1)+b1)
    if train:
        y1=tf.nn.dropout(y1,0.5)

    w2 = get_weight([FC_SIZE,OUT_NODE],regularizer)
    b2 = get_bias(OUT_NODE)
    y = tf.nn.relu(tf.matmul(y1,w2)+b2)
    return y

def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1),dtype=tf.float32)
    if regularizer != None:
        loss = tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool2x2(x):
    #ksize仅描述池化卷积核大小。strides描述滑动步长步长
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# def main():
#     x = tf.constant(1.1,tf.float32,[10,28,28,1])
#     forward(x,True,0.01)
#
# if __name__ == '__main__':
#     main()