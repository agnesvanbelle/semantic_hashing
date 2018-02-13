import tensorflow as tf
from data import make_sample
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


w1 = tf.Variable(tf.truncated_normal([4,12], stddev=0.1), dtype=tf.float32)
w2 = tf.Variable(tf.truncated_normal([12,8], stddev=0.1), dtype=tf.float32)

x = tf.placeholder(tf.float32, shape=[None, 4]) # output: None x 4
hidden_1 = tf.nn.tanh(tf.matmul(x, w1)) # output: None x 12
projection = tf.matmul(hidden_1, w2) # output: 12 x 2
hidden_2 = tf.nn.tanh(projection) 
hidden_3 = tf.nn.tanh(tf.matmul(hidden_2, tf.transpose(w2))) # output: 12 x 12         
hidden_3 = tf.nn.dropout(hidden_3, 0.5)
output = tf.matmul(hidden_3, tf.transpose(w1)) # output: 12 x 4
y = tf.placeholder(tf.float32, shape=[None, 4]) # output: None x 4


loss = tf.reduce_mean(tf.reduce_sum((output - y) * (output - y), 1))
optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
#optimize = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
#optimize = tf.train.MomentumOptimizer(0.01,0.001, use_nesterov=True).minimize(loss)
init = tf.global_variables_initializer()

#print(type(hidden_1), type(x), projection)

def train_network(xsample, ysample, col):
  with tf.Session() as session:
    session.run(init)
    feed_dict = {x : xsample, y:xsample}
    pca = None
    for batch in range (5000):
      session.run(optimize, feed_dict)
      eval_loss, projected_data = session.run([loss, projection], feed_dict=feed_dict)
      
      if batch % 1000 == 0:
        print ('loss: %g' % eval_loss)
        plt.clf()
        #tSne = TSNE(n_components=2, learning_rate=20).fit(projected_data)
        pca = PCA(n_components=2).fit(projected_data)
        projected_data_reduced = pca.transform(projected_data)
        #projected_data_reduced = PCA(n_components=2, whiten=True).fit_transform(projected_data)
        plt.scatter(projected_data_reduced[:,0], projected_data_reduced[:,1], c = col)
        plt.title('batch %d, loss %g' % (batch, eval_loss))
        plt.show(block=False)
        plt.pause(.001)
  
    (xsample_test, ysample_test, col_test) = make_sample(10)
    eval_loss, projected_data = session.run([loss, projection], feed_dict={x: xsample_test, y: ysample_test})
    plt.clf()
    projected_data_reduced = pca.transform(projected_data)
    plt.scatter(projected_data_reduced[:,0], projected_data_reduced[:,1], c = col_test)
    plt.title('test data, loss %g' % (eval_loss))
    plt.show()
  
if __name__=="__main__":
  print("hi")
  (xsample, ysample, col) = make_sample(1000)
  train_network(xsample, ysample, col)
  
  