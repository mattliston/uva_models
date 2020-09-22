import tensorflow as tf
import numpy as np
#import tensorflow_model_optimization as tfmot
import argparse
import json

from mat4py import loadmat


def loss_metric1(y_true, y_pred):
    loss = tf.keras.losses.MeanSquaredError()
    return loss(y_true[:,0], y_pred[:,0])

def loss_metric2(y_true, y_pred):
    loss = tf.keras.losses.MeanSquaredError()
    return loss(y_true[:,1], y_pred[:,1])

def loss_metric3(y_true, y_pred):
    loss = tf.keras.losses.MeanSquaredError()
    return loss(y_true[:,2], y_pred[:,2])

def loss_metric4(y_true, y_pred):
    loss = tf.keras.losses.MeanSquaredError()
    return loss(y_true[:,3], y_pred[:,3])

def loss_metric5(y_true, y_pred):
    loss = tf.keras.losses.MeanSquaredError()
    return loss(y_true[:,4], y_pred[:,4])

def loss_metric6(y_true, y_pred):
    loss = tf.keras.losses.MeanSquaredError()
    return loss(y_true[:,5], y_pred[:,5])


crnn_model = tf.keras.models.load_model('saved/uva_padova_crnn.h5',custom_objects={'loss_metric1':loss_metric1, 'loss_metric2':loss_metric2, 'loss_metric3':loss_metric3, 'loss_metric4':loss_metric4,'loss_metric5':loss_metric5,'loss_metric6':loss_metric6})

bilstm_model = tf.keras.models.load_model('saved/uva_padova_crnn.h5',custom_objects={'loss_metric1':loss_metric1, 'loss_metric2':loss_metric2, 'loss_metric3':loss_metric3, 'loss_metric4':loss_metric4,'loss_metric5':loss_metric5,'loss_metric6':loss_metric6})

lstm_model = tf.keras.models.load_model('saved/uva_padova_crnn.h5',custom_objects={'loss_metric1':loss_metric1, 'loss_metric2':loss_metric2, 'loss_metric3':loss_metric3, 'loss_metric4':loss_metric4,'loss_metric5':loss_metric5,'loss_metric6':loss_metric6})


print(crnn_model.summary())
print(bilstm_model.summary())
print(lstm_model.summary())

