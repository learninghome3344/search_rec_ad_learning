'''
# Time   : 202/12/20 22:55
# Author : yanbingjian
# File   : model.py
'''

import tensorflow as tf
import tensorflow.keras.backend as K

# model build
class FM_layer(tf.keras.layers.Layer):
    '''
    Args:
    hidden_dim:
        dim of hidden_units
    w_reg:
        regularizer penalty for matrix w, if bigger will impose greater punishment for w, final loss = loss + w_reg * penalty_loss(w)
    v_reg:
        regularizer penalty for matrix v, 
    '''
    def __init__(self, hidden_dim, w_reg, v_reg) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(
            name='w0',
            shape=(1,),
            initializer=tf.zeros_initializer(),
            trainable=True
        )
        self.w = self.add_weight(
            name='w0',
            shape=(input_shape[-1], 1),
            initializer=tf.random_normal_initializer(),
            regularizer=tf.keras.regularizers.l2(self.w_reg),
            trainable=True
        )
        self.v = self.add_weight(
            name='w0',
            shape=(input_shape[-1], self.hidden_dim),
            initializer=tf.zeros_initializer(),
            regularizer=tf.keras.regularizers.l2(self.v_reg),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim))

        linear_part = tf.matmul(inputs, self.w) + self.w0  # (batch_size, 1)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  # (batch_size, hidden_dim)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2)) # (batch_size, hidden_dim)
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True) # (batch_size, 1)
        return tf.nn.sigmoid(linear_part + inter_part)


class FM(tf.keras.Model):
    def __init__(self, hidden_dim, w_reg=1e-4, v_reg=1e-4) -> None:
        super().__init__()
        self.fm = FM_layer(hidden_dim, w_reg, v_reg)
    
    def call(self, inputs):
        output = self.fm(inputs)
        return output