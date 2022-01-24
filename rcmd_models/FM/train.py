'''
# Time   : 202/12/20 22:55
# Author : yanbingjian
# File   : train_lstm.py
'''

from model import FM
from utils import create_criteo_dataset

import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
from sklearn.metrics import accuracy_score

import argparse
parser = argparse.ArgumentParser(description='命令行参数')
parser.add_argument('-k', type=int, help='v_dim', default=8)
parser.add_argument('-w_reg', type=float, help='w正则', default=1e-4)
parser.add_argument('-v_reg', type=float, help='v正则', default=1e-4)
args=parser.parse_args()

if __name__ == '__main__':
    file_path = '../Data/train.txt'
    (X_train, y_train), (X_test, y_test) = create_criteo_dataset(file_path, test_size=0.5)

    k = args.k
    w_reg = args.w_reg
    v_reg = args.v_reg

    model = FM(k, w_reg, v_reg)
    optimizer = optimizers.SGD(0.01)


    # # version1
    # train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    # model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
    #             loss=tf.keras.losses.BinaryCrossentropy(),
    #             metrics=[tf.keras.metrics.BinaryAccuracy()])
    # model.fit(train_dataset, epochs=100)
    # model.evaluate(X_test, y_test)
    # model.summary()

    # pre = model.predict(tf.data.Dataset.from_tensor_slices(X_test).batch(1))
    # pre = [1 if x>0.5 else 0 for x in pre]
    # print("AUC: ", accuracy_score(y_test, pre))


    # version2
    summary_writer = tf.summary.create_file_writer('/home/aaa/tf_projects/rcmd_models/tensorboard')
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pre = model(X_train)
            loss = tf.reduce_mean(losses.binary_crossentropy(y_true=y_train, y_pred=y_pre))
            print(loss.numpy())
        with summary_writer.as_default():
            tf.summary.scalar("loss", loss, step=i)
        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
    #评估
    pre = model(X_test)
    pre = [1 if x>0.5 else 0 for x in pre]
    print("AUC: ", accuracy_score(y_test, pre))





