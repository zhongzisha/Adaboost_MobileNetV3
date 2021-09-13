import sys, os, glob, shutil, pickle, joblib
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
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from datetime import datetime
import mmcv
from mmcv import DictAction
from mmcls.datasets import build_dataset, build_dataloader

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

from sklearn.utils import compute_class_weight, compute_sample_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder  # LabelBinarizer
from tensorflow.keras.datasets import cifar10
import math

import models

"""
CUDA_VISIBLE_DEVICES=1 python test_keras_sklearn.py --sample_weight_type min_norm --base_learning_rate 0.00625 --arch MobileNetV3Small --batch_size 256 --epochs 100 --metrics accuracy

CUDA_VISIBLE_DEVICES=1 python test_keras_sklearn.py --sample_weight_type min_norm --base_learning_rate 0.00625 --arch MobileNetV3Small --batch_size 256 --train_adaboost_cnn 1 --epochs 20 --n_estimators 5 --metrics accuracy

CUDA_VISIBLE_DEVICES=0 python test_keras_sklearn.py --sample_weight_type min_norm --base_learning_rate 0.003125 --arch MobileNetV3Large --batch_size 128 --epochs 100 --metrics accuracy

CUDA_VISIBLE_DEVICES=1 python test_keras_sklearn.py --sample_weight_type min_norm --base_learning_rate 0.003125 --arch MobileNetV3Large --batch_size 128 --train_adaboost_cnn 1 --epochs 20 --n_estimators 5 --metrics accuracy


"""


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


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
    plt.plot(history.history['categorical_accuracy'], color='blue', label='train')
    plt.plot(history.history['val_categorical_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    plt.savefig(filename + '_plot.png')
    plt.close()


# theano doesn't need any seed because it uses numpy.random.seed
def train_CNN(model_func, X_train=None, y_train=None, epochs=None, batch_size=None,
              X_test=None, y_test=None, seed=100, input_shape=(32, 32, 3), num_classes=10,
              base_learning_rate=0.016, sample_weight_type=None):
    ######ranome seed
    np.random.seed(seed)
    # set_random_seed(seed)
    tensorflow.random.set_seed(seed)

    model = model_func(input_shape=input_shape,
                       num_classes=num_classes)
    # base_learning_rate *= 10
    opt = RMSprop(learning_rate=base_learning_rate,
                  rho=0.9,
                  momentum=0.9,
                  epsilon=0.0038,
                  decay=1e-5)  # 0.0316
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt,  # Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', lr_metric])

    lb = OneHotEncoder(sparse=False)
    y_train_b = y_train.reshape(len(y_train), 1)
    y_train_b = lb.fit_transform(y_train_b)
    y_test_b = y_test.reshape(len(y_test), 1)
    y_test_b = lb.fit_transform(y_test_b)

    # train CNN
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    # set_random_seed(seed)

    # def lr_step_decay(epoch, lr):
    #     drop_rate = 0.997
    #     epochs_drop = 2
    #     return base_learning_rate * math.pow(drop_rate, math.floor(epoch / epochs_drop))
    #
    # lr_callback = tensorflow.keras.callbacks.LearningRateScheduler(lr_step_decay)

    if sample_weight_type == 'ones':
        sample_weight = np.ones(X_train.shape[0]) / X_train.shape[0]
    elif sample_weight_type == 'random':
        sample_weight = np.random.rand(X_train.shape[0])
    elif sample_weight_type == 'min_norm':
        sample_weight = np.random.rand(X_train.shape[0])
        sample_weight_min = np.min(sample_weight)
        if sample_weight_min != 0:
            sample_weight /= sample_weight_min
    elif sample_weight_type == 'min_max_norm':
        sample_weight = np.random.rand(X_train.shape[0])
        sample_weight[sample_weight < 0.1] = 0.1
        sample_weight[sample_weight > 0.9] = 0.9
        sample_weight /= 0.1
    elif sample_weight_type == 'uniform':
        class_weights = compute_class_weight(class_weight=None, classes=np.unique(y_train), y=y_train)
        sample_weight = compute_sample_weight(class_weight=None, y=y_train)
        print('balanced class_weights', class_weights, np.min(class_weights), np.max(class_weights))
        print('balanced sample_weights', sample_weight, np.min(sample_weight), np.max(sample_weight))
    elif sample_weight_type == 'balanced':
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
        print('balanced class_weights', class_weights, np.min(class_weights), np.max(class_weights))
        print('balanced sample_weights', sample_weight, np.min(sample_weight), np.max(sample_weight))
    else:
        sample_weight = None
    if sample_weight is not None:
        print('sample_weight_type', sample_weight_type)
        print('sample_weight', np.min(sample_weight), np.max(sample_weight))
    # fit model
    # history = model.fit(X_train, y_train_b, epochs=epochs, batch_size=batch_size,
    #                     validation_data=(X_test, y_test_b), verbose=1)
    # history = model.fit(X_train, y_train_b, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_b),
    #                     callbacks=[lr_callback], verbose=1, sample_weight=sample_weight)
    history = model.fit(X_train, y_train_b, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test_b),
                        verbose=1, sample_weight=sample_weight)

    # evaluate model
    result = model.evaluate(X_test, y_test_b, verbose=0)
    print('result', result)
    for i, metric_name in enumerate(model.metrics_names):
        # print('> %.3f' % (acc * 100.0))
        print(metric_name, result[i])
    # learning curves
    summarize_diagnostics(history)
    return model


def load_dataset(config_filename, data_filename):
    # # load dataset
    # (trainX, trainY), (testX, testY) = cifar10.load_data()
    # # one hot encode target values
    # # trainY = to_categorical(trainY)
    # # testY = to_categorical(testY)
    # return trainX, trainY, testX, testY

    if os.path.exists(data_filename):
        return

    # config_filename = './ganta_with_tower_state_bs32_forTest.py'
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

    with open(data_filename, 'wb') as fp:
        pickle.dump({"trainX": trainX,
                     "trainY": trainY,
                     "testX": testX,
                     "testY": testY}, fp, protocol=pickle.HIGHEST_PROTOCOL)


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


# Save configuration information
def save_args(args, save_path):
    if not os.path.exists(save_path):
        os.makedirs('%s' % save_path)

    print('Config info -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    with open('%s/args.txt' % save_path, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)
    joblib.dump(args, '%s/args.pkl' % save_path)
    print('\033[0;33m================config infomation has been saved=================\033[0m')


# Record the information printed in the terminal
class Print_Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# call by
# sys.stdout = Logger(os.path.join(save_path,'test_log.txt'))


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="MobileNetV3Small")
    parser.add_argument("--dataset_name", type=str, default="ganta")
    parser.add_argument("--train_adaboost_cnn", type=bool, default=0)
    parser.add_argument("--sample_weight_type", type=str, default=None)
    parser.add_argument("--base_learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_estimators", type=int, default=5)

    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
             '"accuracy", "precision", "recall", "f1_score", "support" for single '
             'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
             'multi-label dataset')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be parsed as a dict metric_options for dataset.evaluate()'
             ' function.')
    return parser.parse_args()


def main():
    # arch = sys.argv[1]
    # dataset_name = sys.argv[2]
    # train_adaboost_cnn = int(sys.argv[3])
    # sample_weight_type = sys.argv[4]
    args = get_args()
    log_dir = os.path.join('logs', args.dataset_name, args.arch, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sys.stdout = Print_Logger(os.path.join(log_dir, 'log.txt'))

    save_args(args, log_dir)
    for filename in glob.glob('*.py'):
        if '_bak' not in filename:
            shutil.copyfile(filename, os.path.join(log_dir, os.path.basename(filename)))

    trainX, trainY, testX, testY = None, None, None, None
    if args.dataset_name == 'ganta':
        input_shape = (224, 224, 3)
        # load dataset
        config_filename = './ganta_with_tower_state_bs32_forTest.py'
        data_filename = './data.pickle'  # os.path.join(log_dir, 'data.pickle')
        if 'MobileNetV3' in args.arch:
            config_filename = './ganta_with_tower_state_bs32_forTest_noNorm.py'
            data_filename = './data_noNorm.pickle'  # os.path.join(log_dir, 'data.pickle')
        load_dataset(config_filename, data_filename)
        with open(data_filename, 'rb') as fp:
            data = pickle.load(fp)
        trainX = data['trainX']
        trainY = data['trainY']
        testX = data['testX']
        testY = data['testY']
        # prepare pixel data
        # trainX, testX = prep_pixels(trainX, testX)
        # define model

        print('trainX shape', trainX.shape)
        print('trainY shape', trainY.shape)
        print('testX shape', testX.shape)
        print('testY shape', testY.shape)

        if trainX.shape[0] / args.batch_size != 0:
            extra_num = int(np.ceil(trainX.shape[0] / args.batch_size) * args.batch_size - trainX.shape[0])
            trainX = np.concatenate([trainX, trainX[:extra_num, :]], axis=0)
            trainY = np.concatenate([trainY, trainY[:extra_num]], axis=0)

        print('trainX shape', trainX.shape)
        print('trainY shape', trainY.shape)
        print('testX shape', testX.shape)
        print('testY shape', testY.shape)
    elif args.dataset_name == 'cifar10':
        (trainX, trainY), (testX, testY) = cifar10.load_data()
        input_shape = (32, 32, 3)
        batch_size = 32
    else:
        print('wrong dataset')
        sys.exit(-1)

    classes = np.unique(trainY)
    num_classes = len(classes)
    # base_learning_rate = 0.016
    # if args.arch == 'MobileNetV3Small':
    #     batch_size = 256
    #     base_learning_rate = 0.00625
    # elif args.arch == 'MobileNetV3Large':
    #     batch_size = 128
    #     base_learning_rate = 0.003125

    if args.train_adaboost_cnn:
        # # # # # ###Adaboost+CNN:

        from multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN

        # n_estimators = 5
        # epochs = 20
        bdt_real_test_CNN = Ada_CNN(
            base_estimator=getattr(models, args.arch)(input_shape=input_shape, num_classes=num_classes),
            n_estimators=args.n_estimators,
            learning_rate=0.01,
            epochs=args.epochs,
            log_dir=log_dir,
            classes=classes,
            base_learning_rate=args.base_learning_rate,
            sample_weight_type=args.sample_weight_type
        )

        bdt_real_test_CNN.fit(trainX, trainY, args.batch_size)
        test_real_errors_CNN = bdt_real_test_CNN.estimator_errors_[:]

        y_pred_CNN = bdt_real_test_CNN.predict(trainX)
        print('\n Training accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
            accuracy_score(bdt_real_test_CNN.predict(trainX), trainY)))

        y_pred_CNN = bdt_real_test_CNN.predict(testX)
        print('\n Testing accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
            accuracy_score(bdt_real_test_CNN.predict(testX), testY)))

        y_pred_prob = bdt_real_test_CNN.predict_proba(testX)
        # import pdb
        # pdb.set_trace()
        with open(os.path.join(log_dir, 'adaboost_cnn_val_results.pkl'), 'wb') as fp:
            pickle.dump(y_pred_prob, fp)

    else:
        print('log_dir', log_dir)
        cnn_model = train_CNN(model_func=getattr(models, args.arch),  # models.VGG_Block_3_with_Dropout_BN,
                          X_train=trainX,
                          y_train=trainY,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          X_test=testX,
                          y_test=testY,
                          seed=seed,
                          input_shape=input_shape,
                          num_classes=num_classes,
                          base_learning_rate=args.base_learning_rate)
        y_pred_prob = cnn_model.predict(testX, batch_size=args.batch_size, verbose=1)
        # import pdb
        # pdb.set_trace()
        with open(os.path.join(log_dir, 'cnn_val_results.pkl'), 'wb') as fp:
            pickle.dump(y_pred_prob, fp)

    # config_filename = './ganta_with_tower_state_bs32_forTest.py'
    cfg = mmcv.Config.fromfile(config_filename)
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)
    outputs = [y_pred_prob[i] for i in range(len(y_pred_prob))]

    results = {}
    if args.metrics:
        eval_results = dataset.evaluate(outputs, args.metrics,
                                        args.metric_options)
        results.update(eval_results)
        for k, v in eval_results.items():
            print(f'\n{k} : {v:.2f}')

    out_filename = os.path.join(log_dir, 'val_results.pkl')
    scores = np.vstack(outputs)
    pred_score = np.max(scores, axis=1)
    pred_label = np.argmax(scores, axis=1)
    pred_class = [dataset.CLASSES[lb] for lb in pred_label]
    results.update({
        'class_scores': scores,
        'pred_score': pred_score,
        'pred_label': pred_label,
        'pred_class': pred_class
    })
    print(f'\ndumping results to {out_filename}')
    mmcv.dump(results, out_filename)

    filenames = list()
    for info in dataset.data_infos:
        if info['img_prefix'] is not None:
            filename = os.path.join(info['img_prefix'],
                                    info['img_info']['filename'])
        else:
            filename = info['img_info']['filename']
        filenames.append(filename)
    gt_labels = list(dataset.get_gt_labels())
    gt_classes = [dataset.CLASSES[x] for x in gt_labels]

    # load test results
    outputs = mmcv.load(out_filename)
    outputs['filename'] = filenames
    outputs['gt_label'] = gt_labels
    outputs['gt_class'] = gt_classes

    print(dataset.CLASSES)
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(gt_labels, outputs['pred_label']))
    print(classification_report(gt_labels, outputs['pred_label'], target_names=dataset.CLASSES))


if __name__ == '__main__':
    main()
