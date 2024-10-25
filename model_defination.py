from keras.callbacks import LambdaCallback, Callback
from keras import layers
from keras.layers import Dense, Bidirectional,Average,Input,InputLayer
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, CuDNNLSTM, Reshape, Lambda
from keras.layers import Conv1D, MaxPooling1D,AveragePooling1D, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
from keras.optimizers import RMSprop
from keras.models import Model
import numpy as np
import random
from utils.attention import AttentionWithContext
import sys
import io
import datetime
import math
import keras
from keras.models import load_model
from keras import backend as K
from itertools import product
from Bio import SeqIO
from keras.callbacks import EarlyStopping
import nni
from nni.nas.tensorflow.mutables import LayerChoice, InputChoice
from nni.algorithms.nas.tensorflow.classic_nas import get_and_apply_next_architecture

def int_shape(x):
    """Returns the shape of a Keras tensor or a Keras variable as a tuple of
    integers or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        return None

def permute_dimensions(x, pattern):
    """Transpose dimensions.

    pattern should be a tuple or list of
    dimension indices, e.g. [0, 2, 1].
    """
    pattern = tuple(pattern)
    y = x.dimshuffle(pattern)
    if hasattr(x, '_keras_shape'):
        y._keras_shape = tuple(np.asarray(x._keras_shape)[list(pattern)])
    return y

def cross_entropy_loss(y_true, y_pred):
    return 1 / np.log(2) * K.categorical_crossentropy(y_true, y_pred)

def seq_crossentropy(y_true, y_pred):
    # shape of y_true: (None, )
    # print(y_true.shape, y_pred.shape)
    # output_dimensions = list(range(len(int_shape(y_pred))))
    # if axis != -1 and axis != output_dimensions[-1]:
    #     permutation = output_dimensions[:axis]
    #     permutation += output_dimensions[axis + 1:] + [axis]
    #     output = permute_dimensions(output, permutation)
    #     target = permute_dimensions(target, permutation)
    loss = 0
    seq_len = y_pred.shape[1]
    for i in range(seq_len):
        y_t = y_true[:,i,:]
        y_p = y_pred[:,i,:]
        loss += 1 / np.log(2) * K.categorical_crossentropy(y_t, y_p)
    return loss
    # return 1 / np.log(2) * K.categorical_crossentropy(y_true, y_pred, axis=1)

def cate_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)

# def sparse_cate_crossentropy(y_true, y_pred):
#     return K.sparse_categorical_crossentropy(y_true, y_pred)

def poisson(y_true, y_pred):
    return keras.losses.poisson(y_true, y_pred)

def kld(y_true, y_pred):
    return keras.losses.kullback_leibler_divergence(y_true, y_pred)

def hinge(y_true, y_pred):
    return keras.losses.hinge(y_true, y_pred)

def square_hinge(y_true, y_pred):
    return keras.losses.squared_hinge(y_true, y_pred)

def cate_hinge(y_ture, y_pred):
    return keras.losses.categorical_hinge(y_ture, y_pred)

def deepdna(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    model = Sequential()
    model.add(InputLayer(input_shape=(context_length, char_num)))
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=3, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(LSTM(256,return_sequences=True, trainable=params['trainable']))
    model.add(Dropout(0.2, trainable=params['trainable']))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(representation_length))
    model.add(Activation('softmax'))
    return model

def cui2020(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=(context_length ,char_num)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))
    model.add(AttentionWithContext())
    model.add(Activation('relu'))
    model.add(Dense(representation_length))
    model.add(Activation('softmax'))
    return model

def deepdna_multiout(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    # representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    predict_base_num = params['predict_base_num']

    model = Sequential()
    model.add(InputLayer(input_shape=(context_length, char_num)))
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=3, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(Bidirectional(LSTM(64, return_sequences=True, trainable=params['trainable']), trainable=params['trainable']))
    #model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.2, trainable=params['trainable']))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(predict_base_num*char_num))
    model.add(Dropout(0.1))
    model.add(Reshape((predict_base_num, char_num)))
    model.add(Activation('softmax'))
    return model

def cui2020_multiout(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    predict_base_num = params['predict_base_num']

    representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=(context_length ,char_num)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))
    model.add(AttentionWithContext())
    model.add(Activation('relu'))
    # model.add(Dense(representation_length))
    model.add(Dense(predict_base_num*char_num))
    model.add(Dropout(0.1))
    model.add(Reshape((predict_base_num, char_num)))
    model.add(Activation('softmax'))
    return model

def LSTM_multi(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    predict_base_num = params['predict_base_num']

    model = Sequential()
    model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
    model.add(CuDNNLSTM(32, stateful=False, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(predict_base_num*char_num, activation='softmax'))
    model.add(Reshape((predict_base_num, char_num)))
    return model

def lec_model(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    model = Sequential()
    model.add(InputLayer(input_shape=(context_length, char_num)))
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=2, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(Conv1D(filters=1024,
                     kernel_size=12,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=2, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(LSTM(512,return_sequences=True, trainable=params['trainable']))
    model.add(Dropout(0.2, trainable=params['trainable']))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(representation_length))
    model.add(Activation('softmax'))
    return model

def lec_model_multiout(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    # representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    predict_base_num = params['predict_base_num']

    model = Sequential()
    model.add(InputLayer(input_shape=(context_length, char_num)))
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=2, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(Conv1D(filters=1024,
                     kernel_size=12,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=2, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(LSTM(512,return_sequences=True, trainable=params['trainable']))
    model.add(Dropout(0.2, trainable=params['trainable']))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(predict_base_num*char_num))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Reshape((predict_base_num, char_num)))
    model.add(Activation('softmax'))

    return model

def lec_model_multiout_cudnn(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    # representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    predict_base_num = params['predict_base_num']

    model = Sequential()
    model.add(InputLayer(input_shape=(context_length, char_num)))
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=2, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(Conv1D(filters=1024,
                     kernel_size=12,
                     padding='valid',
                     activation='relu',
                     strides=1, trainable=params['trainable']))
    model.add(MaxPooling1D(pool_size=2, trainable=params['trainable']))
    model.add(Dropout(0.1, trainable=params['trainable']))
    model.add(CuDNNLSTM(512,return_sequences=True, trainable=params['trainable']))
    model.add(Dropout(0.2, trainable=params['trainable']))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(predict_base_num*char_num))
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Reshape((predict_base_num, char_num)))
    model.add(Activation('softmax'))

    return model

def cui2020_modify(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    model = Sequential()
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1,input_shape=(context_length ,char_num)))
    model.add(MaxPooling1D(pool_size=3))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(256, stateful=False, return_sequences=True)))
    model.add(Bidirectional(LSTM(256)))
    # model.add(AttentionWithContext())
    model.add(Activation('relu'))
    model.add(Dense(representation_length))
    model.add(Activation('softmax'))
    return model

def deepdna_modify(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    model = Sequential()
    model.add(InputLayer(input_shape=(context_length, char_num)))
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     dilation_rate=1,
                     strides=1))

    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(representation_length))
    model.add(Activation('softmax'))
    return model
    

def deepdna_cam(params):
    context_length = params['context_length']
    char_num = len(params['chars'])
    # representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    representation_length = int(math.pow(len(params['chars']), params['predict_base_num']))
    model = Sequential()

    model.add(InputLayer(input_shape=(context_length, char_num)))
    model.add(Conv1D(filters=1024,
                     kernel_size=24,
                     trainable=True,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(Lambda(cam))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.1))
    model.add(LSTM(256,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(representation_length))
    model.add(Activation('softmax'))
    return model

def cam(x):
    # raise NotImplementedError
    batch, size, channel = x.get_shape().as_list()

    gamma = K.variable(np.array([0]), dtype='float32',name='gamma')

    proj_query = Reshape((size, channel))(x)
    proj_key = Reshape((size, channel))(x)
    proj_value = Reshape((size, channel))(x)

    proj_key = K.permute_dimensions(proj_key, (0,2,1))
    energy = K.batch_dot(proj_key, proj_query)

    attention = K.softmax(energy)
    out = K.batch_dot(proj_value, attention)
    #   out = Reshape((height, width, channel))(out)

    out = gamma * out + x
    return out