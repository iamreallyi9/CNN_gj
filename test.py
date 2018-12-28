
import tensorflow as tf

A = [ [ [1,2,3,4],[4,5,6,7],[7,8,9,8]  ], [ [1,4,7,1],[2,5,8,2],[3,6,9,5]  ]]
B = [[1,2,3,4,4,2],[1,2,3,4,5,6]]
x = tf.split(A,2,2)

with tf.Session() as sess:
    c = sess.run(x)
    print(sess.run(tf.shape(A)))
    for ele in c:
        print(ele)