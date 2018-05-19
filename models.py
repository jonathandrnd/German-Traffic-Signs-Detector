__author__ = "Jonathan Durand"
__email__ = "jonathan.drnd@gmail.com"

import cv2
import time,random
import os,shutil
import urllib.request
import zipfile
import click
import matplotlib.pyplot as plt
import numpy as np
import pickle,sys
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from PIL import Image
from tensorflow.contrib.layers import flatten
from app import *

ROWS=32
COLS=32
NUM_CLASSES=43

def logistic_regression_scikit(dir_train_or_test,istraining,model=-1,Xinfer=None):
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("models/model1"):
        os.makedirs("models/model1")
    if not os.path.exists("models/model1/saved"):
        os.makedirs("models/model1/saved")

    filename = 'models/model1/saved/model1.sav'
    if istraining==True:
        Xtrain, Ytrain = read_train_test_from_directory(dir_train_or_test)
        #Xtrain, Ytrain = data_augmentation(Xtrain,Ytrain)
        Xtrain, _ = preprocess_data(Xtrain, Ytrain)
        Xtrain = Xtrain.reshape([Xtrain.shape[0], -1])
        model_logistic = LogisticRegression()
        print("Training Logistic Regression Scikit")
        model_logistic.fit(Xtrain, Ytrain)
        pickle.dump(model_logistic, open(filename, 'wb'))
        print("Saved Model")
    else:
        if model==-1:
            Xtest, Ytest = read_train_test_from_directory(dir_train_or_test)
            Xtest, _ = preprocess_data(Xtest, Ytest)
            Xtest = Xtest.reshape([Xtest.shape[0], -1])
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.score(Xtest, Ytest)
            print("score: ",result)
        else:
            loaded_model = pickle.load(open(filename, 'rb'))
            predictions = loaded_model.predict(Xinfer)
            return predictions

def logistic_regression_tensorflow(dir_train_or_test,istraining,model=-1,Xinfer=None):
    # hyperparameters
    learning_rate = 0.01
    num_epochs = 40000
    display_step = 200

    with tf.name_scope("Declaring_placeholder"):
        # X is placeholder Images
        X = tf.placeholder(tf.float32, [None, 32*32])
        # y is placeholder labels
        y = tf.placeholder(tf.float32, [None, 43])

    with tf.name_scope("Declaring_variables"):
        # W is our weights. This will update during training time
        W = tf.Variable(tf.zeros([32*32, 43]))
        # b is our bias. This will also update during training time
        b = tf.Variable(tf.zeros([43]))

    with tf.name_scope("Declaring_functions"):
        # our prediction function
        y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b))

    with tf.name_scope("calculating_cost"):
        # calculating cost
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=1))

    with tf.name_scope("declaring_gradient_descent"):
        # optimizer
        # we use gradient descent for our optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("models/model2"):
        os.makedirs("models/model2")
    if not os.path.exists("models/model2/saved"):
        os.makedirs("models/model2/saved")

    if istraining==True:
        Xtrain, Ytrain = read_train_test_from_directory(dir_train_or_test)
        #Xtrain, Ytrain = data_augmentation(Xtrain,Ytrain)
        Xtrain, Ytrain = preprocess_data(Xtrain, Ytrain)
        Xtrain = Xtrain.reshape([Xtrain.shape[0], -1])
        print("Training Logistic Regression Tensorflow ...")

        with tf.name_scope("starting_tensorflow_session"):
            with tf.Session() as sess:
                # initialize all variables
                sess.run(tf.global_variables_initializer())
                for epoch in range(num_epochs):
                    cost_in_each_epoch = 0
                    # let's start training
                    _, c = sess.run([optimizer, cost], feed_dict={X: Xtrain, y: Ytrain})
                    cost_in_each_epoch += c
                    # you can uncomment next two lines of code for printing cost when training
                    if (epoch+1) % display_step == 0:
                        print("Epoch {} of {} cost {:.5f} ".format(epoch + 1,num_epochs,c))
                    pathglobal=os.getcwd()+"/models/model2/saved/modeltf.ckpt"
                    if (epoch + 1) % 1000 == 0:
                        saver.save(sess,pathglobal,global_step=epoch+1)
                saver.save(sess, pathglobal, global_step=num_epochs)
                print("Saved Model")

    else:
        if model==-1:
            Xtest, Ytest = read_train_test_from_directory(dir_train_or_test)
            Xtest, Ytest = preprocess_data(Xtest, Ytest)
            Xtest = Xtest.reshape([Xtest.shape[0], -1])

            with tf.Session() as sess:
                pathglobal = os.getcwd() + "/models/model2/saved/"
                saver.restore(sess, tf.train.latest_checkpoint(pathglobal))
                accuracy = sess.run(accuracy_operation, feed_dict={X: Xtest, y: Ytest})
                print("Test Accuracy = {:.3f}".format(accuracy))
        else:
            with tf.Session() as sess:
                pathglobal = os.getcwd() + "/models/model2/saved/"
                saver.restore(sess, tf.train.latest_checkpoint(pathglobal))
                _=np.zeros([Xinfer.shape[0],NUM_CLASSES])
                predictions = sess.run(y_, feed_dict={X: Xinfer, y: _})
                predictions=np.array(predictions)
                return np.argmax(predictions,1)

    return ;



def LeNet(x):
    #hyperparameters
    mu = 0
    sigma = 0.1

    #Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    image_depth =1
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_depth, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Avg Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    accuracy = sess.run(accuracy_operation, feed_dict={X_data, y_data})
    return accuracy / num_examples

def lenet_tensorflow(dir_train_or_test,istraining,model=-1,Xinfer=None):
    EPOCHS = 500
    image_depth=1
    x = tf.placeholder(tf.float32, [None, 32, 32, image_depth],name="x")
    y = tf.placeholder(tf.float32, [None,43],name="y")

    rate = 0.003
    logits = LeNet(x)

    varss = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in varss
                       if '_b' not in v.name]) * 0.0001
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy) + lossL2
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    cost_arr = []

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("models/model3"):
        os.makedirs("models/model3")
    if not os.path.exists("models/model3/saved"):
        os.makedirs("models/model3/saved")

    if istraining==True:
        Xtrain, Ytrain = read_train_test_from_directory(dir_train_or_test)
        #Xtrain, Ytrain = data_augmentation(Xtrain,Ytrain)
        Xtrain, Ytrain = preprocess_data(Xtrain, Ytrain)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("Training...")
            for i in range(EPOCHS):
                to, cost = sess.run([training_operation, loss_operation], feed_dict={x:Xtrain, y:Ytrain})
                cost_arr.append(cost)
                pathglobal = os.getcwd() + "/models/model3/saved/lenet.ckpt"
                if (i+1)%10==0:
                    print("EPOCH; {} of {}; Loss; {:.5f}".format(i + 1, EPOCHS, cost))
                if i%100==0:
                    saver.save(sess, pathglobal, global_step=i + 1)
            saver.save(sess, pathglobal, global_step=i+1)
            print("Model saved")
    else:
        if model==-1:
            with tf.Session() as sess:
                pathglobal = os.getcwd() + "/models/model3/saved/"
                saver.restore(sess, tf.train.latest_checkpoint(pathglobal))
                Xtest, Ytest = read_train_test_from_directory(dir_train_or_test)
                Xtest, Ytest = preprocess_data(Xtest, Ytest)
                accuracy = sess.run(accuracy_operation, feed_dict={x: Xtest, y: Ytest})
                print("Test Accuracy = {:.3f}".format(accuracy))
        else:
            with tf.Session() as sess:
                pathglobal = os.getcwd() + "/models/model3/saved/"
                saver.restore(sess, tf.train.latest_checkpoint(pathglobal))
                _ =np.zeros([Xinfer.shape[0],43])
                accuracy = sess.run(logits, feed_dict={x: Xinfer, y: _})
                return np.argmax(accuracy,1)
