#train the model by running training_op


import tensorflow as tf
import le_backward
from tensorflow.examples.tutorials.mnist import input_data
import le_backward
import le_forward
import time
import numpy as np

TEST_SEC = 5

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,(mnist.test.num_examples,le_forward.IMAGE_SIZE,le_forward.IMAGE_SIZE,le_forward.NUM_CHANNELS))
        y_ = tf.placeholder(tf.float32,(mnist.test.num_examples,le_forward.OUT_NODE))
        y = le_forward.forward(x,False,None)

        #定义滑动平均，test时不包含global_step

        ema = tf.train.ExponentialMovingAverage(le_backward.MOVING_AVERAGE_DECAY)
        #Returns a map of names to `Variables` to restore
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        corrcet_prediction = tf.equal(tf.arg_max(y,1),(tf.arg_max(y_,1)))
        accuracy = tf.reduce_mean(tf.cast(corrcet_prediction,tf.float32))

        for j in range(5):
            with tf.Session() as sess :
                ckpt = tf.train.get_checkpoint_state(le_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped = np.reshape(mnist.test.images,
                                          (mnist.test.num_examples,
                                           le_forward.IMAGE_SIZE,
                                           le_forward.IMAGE_SIZE,
                                           le_forward.NUM_CHANNELS))

                    accuary_score = sess.run(accuracy, feed_dict={x: reshaped, y_: mnist.test.labels})
                    print("轮数为 %s 准确率为 %s " % (global_step, accuary_score))

                else:
                    print("error!!!@@@")
                    return

            time.sleep(TEST_SEC)



def main():
    mnist = input_data.read_data_sets('./data',one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()