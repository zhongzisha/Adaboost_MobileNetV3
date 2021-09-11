import sys,os,pickle
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# class_names = ('nontower', 'normal', 'jieduan', 'wanzhe')
#
# train_data_root = '/share/home/zhongzisha/datasets/ganta_patch_classification/train'
# val_data_root = '/share/home/zhongzisha/datasets/ganta_patch_classification/val'
# # train_ds = image_dataset_from_directory(directory=train_data_root, class_names=class_names)
# # val_ds = image_dataset_from_directory(directory=val_data_root, class_names=class_names)
#
# dgen_train = ImageDataGenerator(rescale=1./255,
#                                 zoom_range=0.2,
#                                 horizontal_flip=True,
#                                 vertical_flip=True)
# dgen_val = ImageDataGenerator(rescale=1./255)
# train_generator = dgen_train.flow_from_directory(train_data_root,
#                                                  target_size=(256, 256),
#                                                  batch_size=32,
#                                                  class_mode="sparse")
# val_generator = dgen_val.flow_from_directory(val_data_root,
#                                              target_size=(256, 256),
#                                              batch_size=32,
#                                              class_mode="sparse")
#
# print(train_generator.class_indices)
# print(train_generator.image_shape)

# #####randome seed:
# seed = 100
seed = 50
np.random.seed(seed)
# TensorFlow has its own random number generator
# from tensorflow import set_random_seed
# set_random_seed(seed)
import tensorflow

gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

tensorflow.random.set_seed(seed)
# # ####
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder  # LabelBinarizer
from tensorflow.keras.datasets import cifar10
import math

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
              X_test=None, y_test=None, seed=100, input_shape=(32, 32, 3), num_classes=10):
    ######ranome seed
    np.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)

    model = model_func(input_shape=input_shape,
                       num_classes=num_classes)

    lb = OneHotEncoder(sparse=False)
    y_train_b = y_train.reshape(len(y_train), 1)
    y_train_b = lb.fit_transform(y_train_b)
    y_test_b = y_test.reshape(len(y_test), 1)
    y_test_b = lb.fit_transform(y_test_b)

    # train CNN
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    # set_random_seed(seed)

    def lr_step_decay(epoch, lr):
        drop_rate = 0.997
        epochs_drop = 2
        return 0.008 * math.pow(drop_rate, math.floor(epoch / epochs_drop))

    lr_callback = tensorflow.keras.callbacks.LearningRateScheduler(lr_step_decay)

    # fit model
    # history = model.fit(X_train, y_train_b, epochs=epochs, batch_size=batch_size,
    #                     validation_data=(X_test, y_test_b), verbose=1)
    history = model.fit(X_train, y_train_b, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_b),
                        callbacks=[lr_callback], verbose=1)

    # evaluate model
    _, acc = model.evaluate(X_test, y_test_b, verbose=0)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)


def load_dataset():
    # # load dataset
    # (trainX, trainY), (testX, testY) = cifar10.load_data()
    # # one hot encode target values
    # # trainY = to_categorical(trainY)
    # # testY = to_categorical(testY)
    # return trainX, trainY, testX, testY
    import mmcv
    from mmcls.datasets import build_dataset, build_dataloader

    config_filename = './ganta_with_tower_state_bs32_forTest.py'
    cfg = mmcv.Config.fromfile(config_filename)
    train_dataset = build_dataset(cfg.data.train)
    train_data_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True,
        round_up=True)

    cfg.data.val.test_mode = True
    val_dataset = build_dataset(cfg.data.val)

    print(train_dataset.CLASSES)
    print(val_dataset.CLASSES)
    print(len(train_dataset))
    print(len(val_dataset))

    trainX = []
    trainY = []
    prog_bar = mmcv.ProgressBar(len(train_data_loader))
    for i, data in enumerate(train_data_loader):
        # print(data.keys())
        # print(type(data['img']), data['img'].shape)
        # print(type(data['gt_label']), data['gt_label'].shape)
        # break
        trainX.append(data['img'])
        trainY.append(data['gt_label'])

        # batch_size = data['img'].size(0)
        if i == len(train_data_loader):
            break
        prog_bar.update()

    trainX = np.concatenate(trainX)
    trainY = np.concatenate(trainY)
    print(trainX.shape, trainY.shape)

    # import pdb
    # pdb.set_trace()

    prog_bar = mmcv.ProgressBar(len(val_dataset))
    testX = []
    testY = []
    for idx, data in enumerate(val_dataset):
        # print(data.keys())
        # print(type(data['img']), data['img'].shape)
        # break
        testX.append(data['img'])
        testY.append(data['gt_label'])
        prog_bar.update()

    testX = np.stack(testX)
    testY = np.stack(testY)
    print(trainX.shape, trainY.shape)

    # train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
    # val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
    #
    # train_ds_X, train_ds_y = train_ds.as_numpy_iterator()
    # print(len(train_ds_X), len(train_ds_y))

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
    arch = sys.argv[1]
    dataset_name = sys.argv[2]
    train_adaboost_cnn = int(sys.argv[3])
    log_dir = os.path.join('logs', dataset_name, arch)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    trainX, trainY, testX, testY = None, None, None, None
    if dataset_name == 'ganta':
        batch_size = 256
        input_shape = (224, 224, 3)
        # load dataset
        data_filename = './data.pickle'   # os.path.join(log_dir, 'data.pickle')
        if os.path.exists(data_filename):
            with open(data_filename, 'rb') as fp:
                data = pickle.load(fp)
            trainX = data['trainX']
            trainY = data['trainY']
            testX = data['testX']
            testY = data['testY']
        else:
            trainX, trainY, testX, testY = load_dataset()
            with open(data_filename, 'wb') as fp:
                pickle.dump({"trainX": trainX,
                             "trainY": trainY,
                             "testX": testX,
                             "testY": testY}, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # prepare pixel data
        # trainX, testX = prep_pixels(trainX, testX)
        # define model

        print('trainX shape', trainX.shape)
        print('trainY shape', trainY.shape)
        print('testX shape', testX.shape)
        print('testY shape', testY.shape)

        if trainX.shape[0] / batch_size != 0:
            extra_num = int(np.ceil(trainX.shape[0] / batch_size) * batch_size - trainX.shape[0])
            trainX = np.concatenate([trainX, trainX[:extra_num, :]], axis=0)
            trainY = np.concatenate([trainY, trainY[:extra_num]], axis=0)

        print('trainX shape', trainX.shape)
        print('trainY shape', trainY.shape)
        print('testX shape', testX.shape)
        print('testY shape', testY.shape)
    elif dataset_name == 'cifar10':
        (trainX, trainY), (testX, testY) = cifar10.load_data()
        input_shape = (32, 32, 3)
        batch_size = 32
    else:
        print('wrong dataset')
        sys.exit(-1)

    classes = np.unique(trainY)
    num_classes = len(classes)
    if arch == 'MobileNetV3Small':
        batch_size = 256
    elif arch == 'MobileNetV3Large':
        batch_size = 128

    if train_adaboost_cnn == 1:
        # # # # # ###Adaboost+CNN:

        from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN

        n_estimators = 5
        epochs = 20
        bdt_real_test_CNN = Ada_CNN(
            base_estimator=getattr(models, arch)(input_shape=input_shape, num_classes=num_classes),
            n_estimators=n_estimators,
            learning_rate=0.01,
            epochs=epochs,
            log_dir=log_dir,
            classes=classes
        )

        bdt_real_test_CNN.fit(trainX, trainY, batch_size)
        test_real_errors_CNN = bdt_real_test_CNN.estimator_errors_[:]

        y_pred_CNN = bdt_real_test_CNN.predict(trainX)
        print('\n Training accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
            accuracy_score(bdt_real_test_CNN.predict(trainX), trainY)))

        y_pred_CNN = bdt_real_test_CNN.predict(testX)
        print('\n Testing accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
            accuracy_score(bdt_real_test_CNN.predict(testX), testY)))

    else:

        train_CNN(model_func=getattr(models, arch),    # models.VGG_Block_3_with_Dropout_BN,
                  X_train=trainX,
                  y_train=trainY,
                  epochs=100,
                  batch_size=batch_size,
                  X_test=testX,
                  y_test=testY,
                  seed=seed,
                  input_shape=input_shape,
                  num_classes=num_classes)


if __name__ == '__main__':
    main()


