import numpy as np
import tensorflow as tf
import pickle

# please see the notebook for the detail of preprocessing

DATA_PATH = 'data.npy'
RESULT_PATH = 'result_dict.pkl'
TMP_RESULT = 'tmp_result'
data = np.load(DATA_PATH)

n_in = 150
n_out = 100
L2_reg = 0.0001
learning_rate = 0.001
Epoch = 10


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(initializer=initial, trainable=True, name=name)


def bias_variable(shape, name):
    initial = tf.zeros(shape)
    return tf.get_variable(initializer=initial, dtype=tf.float32, trainable=True, name=name)


def ntn_variable(n_in, n_out, name):
    T = weight_variable(shape=[n_out, n_in, n_in], name="T" + name)
    W1 = weight_variable(shape=[n_out, n_in], name="W" + name + "1")
    W2 = weight_variable(shape=[n_out, n_in], name="W" + name + "2")
    b = bias_variable(shape=[n_out], name="b" + name)
    return T, W1, W2, b


T1, W11, W12, b1 = ntn_variable(n_in, n_out, '1')
T2, W21, W22, b2 = ntn_variable(n_in, n_out, '2')
T3, W31, W32, b3 = ntn_variable(n_out, n_out, '3')
u = tf.Variable(tf.truncated_normal(shape=[n_out], stddev=0.1))
b = tf.Variable(0., dtype=tf.float32)

O1 = tf.placeholder(dtype=tf.float32, shape=[n_in, ], name='O1')
O1F = tf.placeholder(dtype=tf.float32, shape=[n_in, ], name='O1F')
P = tf.placeholder(dtype=tf.float32, shape=[n_in, ], name='P')
O2 = tf.placeholder(dtype=tf.float32, shape=[n_in, ], name='O2')


def R(O, T, A, W1, W2, b):
    bilinear1 = tf.tensordot(O, T, axes=[[0], [1]])
    bilinear = tf.tensordot(bilinear1, A, axes=1)
    linear1 = tf.tensordot(W1, O, axes=1)
    linear2 = tf.tensordot(W2, A, axes=1)
    linear = linear1 + linear2 + b
    R_value = tf.nn.tanh(bilinear + linear)
    norm = tf.norm(T) + tf.norm(W1) + tf.norm(W2)
    return R_value, norm


R1, R1N = R(O1, T1, P, W11, W12, b1)
R2, R2N = R(O2, T2, P, W21, W22, b2)
R3, R3N = R(R1, T3, R2, W31, W32, b3)

R1F, R1FN = R(O1F, T1, P, W11, W12, b1)
R3F, R3FN = R(R1F, T3, R2, W31, W32, b3)
G = tf.nn.tanh(tf.tensordot(u, R3, axes=1) + b)
GF = tf.nn.tanh(tf.tensordot(u, R3F, axes=1) + b)

norm = L2_reg * (R1N + R2N + R3N + tf.norm(u))
loss = tf.maximum(0., (1. - G + GF))
train = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss + norm)

O1_ndarray = data[:, 0, :]
P_ndarray = data[:, 1, :]
O2_ndarray = data[:, 2, :]
indices = np.arange(O1_ndarray.shape[0])
np.random.shuffle(indices)
O1_corrupted = O1_ndarray[indices]
saved_tensor = tf.Variable(0, dtype=tf.int32, name='saved_tensor', trainable=False)
mean_loss = tf.Variable(0.0, name='mean_loss', trainable=False)

N_sample = O1_ndarray.shape[0]

result = {}
with tf.Session() as sess:
    # if tf.gfile.Exists('tmp'):
    #     tf.gfile.DeleteRecursively('tmp')
    # tf.gfile.MakeDirs('tmp')
    tf.summary.scalar("loss", mean_loss)
    tf.summary.scalar('saved_tensor', saved_tensor)
    sess.run(tf.global_variables_initializer())
    file_writer = tf.summary.FileWriter('tmp', sess.graph)
    merge_op = tf.summary.merge_all()
    total_step = 0
    tmp_result = np.zeros((N_sample, n_out))
    for epoch in range(Epoch):
        set_index = set(range(N_sample))
        t_index = O1_ndarray.shape[0]
        Max_inter = len(set_index) * 4
        iter_num = 0
        O1_corrupted = np.random.randn(N_sample, n_in) + O1_ndarray.mean(axis=0)
        O2_corrupted = np.random.randn(N_sample, n_in) + O2_ndarray.mean(axis=0)
        outLOSS = []
        if len(result.keys()) == N_sample:
            np.save(TMP_RESULT, tmp_result)
            with open(RESULT_PATH, 'wb') as handle:
                pickle.dump(result, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            break
        while len(set_index) > 0 and iter_num <= Max_inter:
            total_step += 1
            iter_num += 1
            index = set_index.pop()
            o1 = O1_ndarray[index]
            p = P_ndarray[index]
            o2 = O2_ndarray[index]
            if total_step % 2 == 0:
                o1f = O1_corrupted[index]
            else:
                o1f = O2_corrupted[index]
            nm, loss_value, g, gf = sess.run([norm, loss, G, GF], feed_dict={O1: o1, O2: o2, P: p, O1F: o1f})
            outLOSS.append(loss_value)
            if loss_value > 0:
                set_index.add(index)
                sess.run(train, feed_dict={O1: o1, O2: o2, P: p, O1F: o1f})
            else:
                cooked_vector = sess.run(R3, feed_dict={O1: o1, O2: o2, P: p, O1F: o1f})
                result[index] = cooked_vector
                tmp_result[index] = cooked_vector
            sess.run(saved_tensor.assign(len(result.keys())))
            sess.run(mean_loss.assign(np.mean(outLOSS)))
            summaries = sess.run(merge_op, feed_dict={O1: o1, O2: o2, P: p, O1F: o1f})
            file_writer.add_summary(summaries, total_step)
            if (len(set_index) % 50 == 0 and t_index != len(set_index)):
                t_index = len(set_index)
                print("There are %f %% left in this %d epoch at iteration %d at total step %d with average cost %f" % (len(set_index) / float(N_sample) * 100, epoch, iter_num, total_step, np.mean(outLOSS)))
                np.save(TMP_RESULT, tmp_result)
                with open(RESULT_PATH, 'wb') as handle:
                    pickle.dump(result, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
            if iter_num % 100 == 0:
                print("There are %f %% left in this %d epoch at iteration %d at total step %d with average cost %f" % (len(set_index) / float(N_sample) * 100, epoch, iter_num, total_step, np.mean(outLOSS)))
                np.save(TMP_RESULT, tmp_result)
                with open(RESULT_PATH, 'wb') as handle:
                    pickle.dump(result, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
