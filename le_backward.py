import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import le_forward
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.05
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"


def backward(mnist):
    #必须指定x，y_占位BATCH_SIZE
    x = tf.placeholder(tf.float32, [BATCH_SIZE, le_forward.IMAGE_SIZE,le_forward.IMAGE_SIZE,le_forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, le_forward.OUT_NODE])
    y = le_forward.forward(x,True,REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    #global_step 在minimize中自动完成加一操作
    #其余地方用到global_step都是来迎合这个节拍时钟

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #定义滑动平均，按照global_step行事
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            resahped_x = np.reshape(xs,(BATCH_SIZE,le_forward.IMAGE_SIZE,le_forward.IMAGE_SIZE,le_forward.NUM_CHANNELS))
            #X必须变形因为卷积层要的是矩阵不是数组，Y不用变形，因为最后还是原来的全连接层输出
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: resahped_x , y_: ys})
            if i % 5 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()


