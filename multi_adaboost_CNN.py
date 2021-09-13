__author__ = 'Xin, Aboozar'

import sys, os
import numpy as np
from numpy.core.umath_tests import inner1d
from tensorflow.keras.models import Sequential
import tensorflow
from sklearn.preprocessing import OneHotEncoder  # LabelBinarizer
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
import gc
import math


# this is function to load weight file for resuming training from last epoch
def checkpoint_function(checkpoint_path):
    # if there is no file being saved in the checkpoint path then routine will go to loop model.fit(X, y, epochs=150,
    # batch_size=1, callbacks=[ckpt_callback]) this signifies that no file is saved in the checkpoint path and let us
    # begin the training from the first epoch.
    if not os.listdir(checkpoint_path):
        return None, None
    # this loop will fetch epochs number in a list
    files_int = list()
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('.')[1])
        files_int.append(epoch)
    # getting the maximum value for an epoch from the list
    # this is reference value and will help to find the file with that value
    max_value = max(files_int)
    # conditions are applied to find the file which has the maximum value of epoch such file would be the last file
    # where the training is stopped and we would like to resume the training from that point.
    final_file = None
    for i in os.listdir(checkpoint_path):
        epoch = int(i.split('.')[1])
        if epoch > max_value:
            pass
        elif epoch < max_value:
            pass
        else:
            final_file = i
            # in the end we will get the epoch as well as file
    return final_file, max_value


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class AdaBoostClassifier(object):
    '''
    Parameters
    -----------
    base_estimator: object
        The base model from which the boosted ensemble is built.

    n_estimators: integer, optional(default=50)
        The maximum number of estimators

    learning_rate: float, optional(default=1)

    algorithm: {'SAMME','SAMME.R'}, optional(default='SAMME.R')
        SAMME.R uses predicted probabilities to update wights, while SAMME uses class error rate

    random_state: int or None, optional(default=None)


    Attributes
    -------------
    estimators_: list of base estimators

    estimator_weights_: array of floats
        Weights for each base_estimator

    estimator_errors_: array of floats
        Classification error for each estimator in the boosted ensemble.

    Reference:
    1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)

    2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/
    scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)

    '''

    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate',
                        'algorithm', 'random_state', 'epochs', 'log_dir',
                        'classes', 'base_learning_rate', 'sample_weight_type',
                        'seed']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        base_estimator = None
        n_estimators = 50
        learning_rate = 1
        algorithm = 'SAMME.R'
        random_state = None
        #### CNN (5)
        epochs = 6
        log_dir = '.'
        classes = ("0", "1")
        base_learning_rate = 0.016
        sample_weight_type = 'none'
        seed = 1024

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')
            ### CNN:
            if 'epochs' in kwargs: epochs = kwargs.pop('epochs')
            if 'log_dir' in kwargs: log_dir = kwargs.pop('log_dir')
            if 'classes' in kwargs: classes = kwargs.pop('classes')
            if 'base_learning_rate' in kwargs: base_learning_rate = kwargs.pop('base_learning_rate')
            if 'sample_weight_type' in kwargs: sample_weight_type = kwargs.pop('sample_weight_type')
            if 'seed' in kwargs: seed = kwargs.pop('seed')

        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.random_state_ = random_state
        # self.estimators_ = [os.path.join(log_dir, 'model_0.h5')]
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        self.estimator_errors_ = np.ones(self.n_estimators_)

        # self.base_estimator_.save(self.estimators_[0])

        self.epochs = epochs
        self.log_dir = log_dir
        self.classes_ = classes
        self.base_learning_rate = base_learning_rate
        self.sample_weight_type = sample_weight_type
        self.seed = seed

        checkpoint_path = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        self.ckpt_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_path,
                                                                   'weights.{epoch:02d}.{categorical_accuracy:.4f}.hdf5'),
                                             monitor='categorical_accuracy')
        self.checkpoint_path = checkpoint_path

        # def lr_step_decay(epoch, lr):
        #     drop_rate = 0.997
        #     epochs_drop = 3
        #     return self.base_learning_rate * math.pow(drop_rate, math.floor(epoch / epochs_drop))
        #
        # self.lr_callback = tensorflow.keras.callbacks.LearningRateScheduler(lr_step_decay)

    def _samme_proba(self, estimator_filename, n_classes, X):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        estimator = load_model(estimator_filename)
        proba = estimator.predict(X, batch_size=self.batch_size)
        tensorflow.keras.backend.clear_session()
        del estimator
        gc.collect()

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes)
                                  * log_proba.sum(axis=1)[:, np.newaxis])

    def _helper(self, estimator_filename, X, classes, w):

        estimator = load_model(estimator_filename)
        result = (estimator.predict(X, batch_size=self.batch_size).argmax(axis=1) == classes).T * w
        del estimator
        tensorflow.keras.backend.clear_session()
        gc.collect()

        return result

    def _helper2(self, estimator_filename, X, w):
        estimator = load_model(estimator_filename)
        result = estimator.predict(X, batch_size=self.batch_size) * w
        tensorflow.keras.backend.clear_session()
        del estimator
        gc.collect()

        return result

    def fit(self, X, y, batch_size):

        ## CNN:
        self.batch_size = batch_size

        #        self.epochs = epochs
        self.n_samples = X.shape[0]
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort

        ############for CNN (2):
        #        yl = np.argmax(y)
        #        self.classes_ = np.array(sorted(list(set(yl))))

        self.n_classes_ = len(self.classes_)
        print('self.n_classes_', self.n_classes_)
        sample_weight = None
        for iboost in range(self.n_estimators_):
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples

            print(iboost, '++' * 20)

            sample_weight, estimator_weight, estimator_error = self.boost(X, y, sample_weight)
            if sample_weight is not None:
                print('sample_weight', sample_weight[:10])

            # early stop
            if estimator_error is None:
                break

            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

            if estimator_error <= 0:
                break

        return self

    def boost(self, X, y, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(X, y, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(X, y, sample_weight)

    def real_boost(self, X, y, sample_weight):

        ######ranome seed
        np.random.seed(self.seed)
        # set_random_seed(seed)
        tensorflow.random.set_seed(self.seed)

        if sample_weight is None:
            sample_weight = np.ones(self.n_samples) / self.n_samples
        max_value = None
        if len(self.estimators_) == 0:
            # Copy CNN to estimator:
            estimator = self.base_estimator_
        else:
            # estimator = deepcopy(self.estimators_[-1])
            estimator, max_value = self.deepcopy_CNN(self.estimators_[-1])  # deepcopy CNN
        if self.random_state_:
            estimator.set_params(random_state=1)
        #        estimator.fit(X, y, sample_weight=sample_weight)
        #################################### CNN (3) binery label:
        # lb=LabelBinarizer()
        # y_b = lb.fit_transform(y)

        opt = RMSprop(learning_rate=self.base_learning_rate,
                      rho=0.9,
                      momentum=0.9,
                      epsilon=0.0038,
                      decay=1e-5)  # 0.0316
        # lr_metric = get_lr_metric(opt)
        estimator.compile(optimizer=opt,  # Adam(learning_rate=0.001),
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])

        print('X', X.shape)
        print('y', y.shape)
        y = y.squeeze()

        lb = OneHotEncoder(sparse=False)
        y_b = y.reshape(len(y), 1)
        y_b = lb.fit_transform(y_b)
        print('self.n_classes_', self.n_classes_)
        print('y_b', y_b.shape)
        print('sample_weight', sample_weight.shape)

        print('fitting estimator')
        sample_weight1 = np.copy(sample_weight)
        if self.sample_weight_type == 'min_norm':
            # sample_weight = np.random.rand(X.shape[0])
            sample_weight_min = np.min(sample_weight)
            print('sample_weight_min', sample_weight_min)
            if sample_weight_min != 0:
                sample_weight1 = sample_weight / sample_weight_min
            else:
                sample_weight1_min = np.min(sample_weight1[np.where(sample_weight1 > 0)])
                sample_weight1[np.where(sample_weight1 == 0)] = sample_weight1_min
                sample_weight1 /= sample_weight1_min
        elif self.sample_weight_type == 'min_max_norm':
            # sample_weight = np.random.rand(X.shape[0])
            if len(np.unique(sample_weight)) > 1:
                sample_weight_min = np.min(sample_weight1)
                sample_weight_max = np.max(sample_weight1)
                ratio = 9.0 / (sample_weight_max - sample_weight_min)
                sample_weight1 -= sample_weight_min
                sample_weight1 = 1 + ratio * sample_weight1  # [xmin, xmax] --> [1, 10]
            else:
                sample_weight1 = np.ones_like(sample_weight)

        if sample_weight1 is not None:
            print('sample_weight_type', self.sample_weight_type)
            print('sample_weight', np.min(sample_weight1), np.max(sample_weight1))

        if max_value is None:
            estimator.fit(X, y_b, sample_weight=sample_weight1, epochs=self.epochs,
                          callbacks=[self.ckpt_callback],  # self.lr_callback
                          batch_size=self.batch_size, verbose=1)
        else:
            estimator.fit(X, y_b, sample_weight=sample_weight1, epochs=self.epochs + max_value,
                          callbacks=[self.ckpt_callback], initial_epoch=max_value,
                          batch_size=self.batch_size, verbose=1)

        print('saving estimator')
        estimator_filename = os.path.join(self.log_dir, 'model_%d.h5' % (len(self.estimators_)))
        estimator.save(estimator_filename)
        ############################################################
        print('predict estimator')
        y_pred = estimator.predict(X, batch_size=self.batch_size, verbose=1)
        tensorflow.keras.backend.clear_session()
        del estimator
        gc.collect()

        ############################################ (4) CNN :
        print('y_pred', y_pred.shape)
        y_pred_l = np.argmax(y_pred, axis=1)
        print('y_pred_l', y_pred_l.shape)
        incorrect = y_pred_l != y
        print('incorrect', len(np.where(incorrect == True)[0]) / len(y))
        print('sample_weight', sample_weight.shape)
        #########################################################
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
        print('estimator_error', estimator_error)

        # import pdb
        # pdb.set_trace()

        # if worse than random guess, stop boosting
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None

        y_predict_proba = y_pred  # estimator.predict(X, batch_size=self.batch_size)

        # repalce zero
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])

        # for sample weight update
        intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) *
                                                              inner1d(y_coding, np.log(
                                                                  y_predict_proba))))  # dot iterate for each row

        # update sample weight
        sample_weight *= np.exp(intermediate_variable)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None

        # normalize sample weight
        sample_weight /= sample_weight_sum

        # append the estimator
        self.estimators_.append(estimator_filename)

        return sample_weight, 1, estimator_error

    def deepcopy_CNN(self, base_estimator0_filename):
        print('invoke deepcopy_CNN')
        # tensorflow.keras.backend.clear_session()
        # ckpt_callback = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss')

        # base_estimator0 = load_model(base_estimator0_filename)

        # # print(base_estimator0.summary())
        #
        # # Copy CNN (self.base_estimator_) to estimator:
        # config = base_estimator0.get_config()
        # # estimator = Models.model_from_config(config)
        # estimator = Sequential.from_config(config)
        # # print('config', config)
        #
        # weights = base_estimator0.get_weights()
        # estimator.set_weights(weights)
        #
        # opt = tensorflow.keras.optimizers.RMSprop(learning_rate=0.016,
        #               rho=0.9,
        #               momentum=0.9,
        #               epsilon=0.0079)  # 0.0316
        # # estimator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # estimator.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        #
        # del base_estimator0
        # tensorflow.keras.backend.clear_session()
        #
        # # estimator.summary()
        #
        # return estimator

        checkpoint_path_file, max_value = checkpoint_function(self.checkpoint_path)
        # this is interesting loop of the code. Code has been divided into conditions here
        # in the if condition if checkpoint is not none, then file and last epcoh is restored
        # Such file is loaded using load_model and training start from that epoch
        if checkpoint_path_file is not None:
            # Load model:
            model = load_model(os.path.join(self.checkpoint_path, checkpoint_path_file))
            # model.fit(X, y, epochs=150, batch_size=1, callbacks=[ckpt_callback], initial_epoch=max_value)
            # if there is no checkpoint_path_file it means no file is saved and hence model will be trained from scratch
            return model, max_value  # estimator
        return None, None

    def predict(self, X):
        classes = self.classes_[:, np.newaxis]
        pred = None

        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            pred = sum(self._samme_proba(estimator, self.n_classes_, X) for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            #            pred = sum((estimator.predict(X) == classes).T * w
            #                       for estimator, w in zip(self.estimators_,
            #                                               self.estimator_weights_))
            ########################################CNN disc
            pred = sum(self._helper(estimator, X, classes, w)
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))
        ###########################################################
        pred /= self.estimator_weights_.sum()
        if self.n_classes_ == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def predict_proba(self, X):
        n_classes = self.n_classes_
        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            proba = sum(self._samme_proba(estimator, self.n_classes_, X)
                        for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
            proba = sum(self._helper2(estimator, X, w)
                        for estimator, w in zip(self.estimators_,
                                                self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
