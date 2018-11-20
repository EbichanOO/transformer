import numpy as np
from chainer import functions as F

def scaled_dot_product_attention(Q, K, V):
    #今回マスクは付けず、またdot関数を使いました
    dump = np.dot(Q, K.T)
    dump = (dump - dump.mean()) / dump.std()
    return np.dot(F.softmax(dump), V)

def linear_trans(x, n_head):
    new_x_shape = x.shape[:-1] + (n_head, x.shape[-1]//n_head)
    A = np.identity(new_x_shape)
    x = x.reshape(*new_x_shape)
    return np.dot(A, x.T), F.transpose(x, (0, 2, 1, 3))

def multi_Head_Attention(Q, K, V):
    Q, L = linear_trans(Q, 2)
    print(Q.shape, L.shape)
    #K = linear_trans(K)
    #V = linear_trans(V)

x = np.ones((5, 30), dtype=float)
multi_Head_Attention(x, 0, 0)