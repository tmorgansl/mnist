# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 08:35:41 2019

@author: tamor
"""

import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

X_train = X_train / 255.
X_test = X_test / 255.

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(), # 28 x 28 -> 784 x 1
        tf.keras.layers.Dense(10, activation=tf.nn.softmax) #  784 x1 -> 10 x1
        ])

# optimizer and loss

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, callbacks=[tf.keras.callbacks.TensorBoard(log_dir='mnist')])

#compute the accuracy for the test set

loss, accuracy = model.evaluate(X_test, y_test)
print('{:.5}'.format(accuracy))