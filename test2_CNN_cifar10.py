__author__ = 'Aboozar'
'''
Reference:
    This is an implementation of the following paper:
    Please cite:
        Taherkhani, Aboozar, Georgina Cosma, and T. M. McGinnity. "AdaBoost-CNN: an adaptive boosting algorithm for convolutional neural networks to classify multi-class imbalanced datasets using transfer learning." Neurocomputing (2020).
    https://www.sciencedirect.com/science/article/pii/S0925231220304379

'''
import sys, os
import numpy as np

# #####randome seed:
# seed = 100
seed = 50
np.random.seed(seed)
# TensorFlow has its own random number generator
# from tensorflow import set_random_seed
# set_random_seed(seed)
import tensorflow
tensorflow.random.set_seed(seed)
# # ####
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder  # LabelBinarizer
from tensorflow.keras.datasets import cifar10

import models


# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


# theano doesn't need any seed because it uses numpy.random.seed
def train_CNN(model_func, X_train=None, y_train=None, epochs=None, batch_size=None,
              X_test=None, y_test=None, seed=100, num_classes=10):
    ######ranome seed
    np.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)

    model = model_func(num_classes=num_classes)

    lb = OneHotEncoder(sparse=False)
    y_train_b = y_train.reshape(len(y_train), 1)
    y_train_b = lb.fit_transform(y_train_b)
    y_test_b = y_test.reshape(len(y_test), 1)
    y_test_b = lb.fit_transform(y_test_b)

    # train CNN
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    # set_random_seed(seed)

    # fit model
    history = model.fit(X_train, y_train_b, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_b),
                        verbose=1)

    # evaluate model
    _, acc = model.evaluate(X_test, y_test_b, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    # trainY = to_categorical(trainY)
    # testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def main():

    batch_size = 64
    num_classes = 10
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # define model

    print('trainX shape', trainX.shape)
    print('trainY shape', trainY.shape)
    print('testX shape', testX.shape)
    print('testY shape', testY.shape)

    if True:
        # # # # # ###Adaboost+CNN:

        from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN

        n_estimators = 5
        epochs = 2
        bdt_real_test_CNN = Ada_CNN(
            base_estimator=models.VGG_Block_3_with_Dropout(num_classes=num_classes),
            n_estimators=n_estimators,
            learning_rate=0.01,
            epochs=epochs)

        bdt_real_test_CNN.fit(trainX, trainY, batch_size)
        test_real_errors_CNN = bdt_real_test_CNN.estimator_errors_[:]

        y_pred_CNN = bdt_real_test_CNN.predict(trainX)
        print('\n Training accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
            accuracy_score(bdt_real_test_CNN.predict(trainX), trainY)))

        y_pred_CNN = bdt_real_test_CNN.predict(testX)
        print('\n Testing accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
            accuracy_score(bdt_real_test_CNN.predict(testX), testY)))

    if False:

        train_CNN(model_func=models.VGG_Block_3_with_Dropout,
                  X_train=trainX, y_train=trainY, epochs=50,
                  batch_size=batch_size, X_test=testX, y_test=testY,
                  seed=seed, num_classes=num_classes)


if __name__ == '__main__':
    main()