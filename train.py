import tensorflow as tf
import numpy as np
#import tensorflow_model_optimization as tfmot
import argparse
import json

from mat4py import loadmat

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='lstm', help='model to be trained (options are crnn, bilstm, lstm)')
parser.add_argument('--prediction_horizon', default=6, type=int, help='prediction horizon')
parser.add_argument('--time_horizon', default=12, type=int, help='time horizon')
parser.add_argument('--resolution', default=10, type=int, help='downscale the data to default 10 min resolution')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='epochs')
parser.add_argument('--training', default=True, help='apply dropout at test time')
parser.add_argument('--batch', default=True, help='batches data for NN')
args = parser.parse_args()

# Data pre-processing

def downscale(data, resolution):

    # 10 min resolution.. (data.shape[0], 3, 1440) -> (data.shape[0], 10, 3, 144).. breaks one 3,1440 length trajectory into ten 3,144 length trajectories
    # Use ~12 timesteps -> 2-5 timesteps (Use ~2 hours to predict 20-50 mins)

    return np.mean(data.reshape(data.shape[0], data.shape[1], int(data.shape[2]/resolution), resolution), axis=3)



def process_data(aligned_data, time_horizon, ph):

    # 10 min resolution.. breaks each (3,144) trajectory into (144-ph-time_horizon,3,time_horizon) samples

    data = np.zeros((aligned_data.shape[0] * (aligned_data.shape[2]-ph-time_horizon), aligned_data.shape[1], time_horizon))
    label = np.zeros((aligned_data.shape[0] * (aligned_data.shape[2]-ph-time_horizon), ph))

    count = 0
    for i in range(aligned_data.shape[0]): # for each sample
        for j in range(aligned_data.shape[2]-ph-time_horizon): # TH length sliding window across trajectory
                data[count] = aligned_data[i,:,j:j+time_horizon]
                label[count] = aligned_data[i,0,j+time_horizon:j+time_horizon+ph]
                count+=1

    return data, label


def load_mpc(time_horizon, ph, resolution, batch): # int, int, int, bool

    # Load train data
    g = np.loadtxt('CGM_prediction_data/glucose_readings_train.csv', delimiter=',')
    c = np.loadtxt('CGM_prediction_data/meals_carbs_train.csv', delimiter=',')
    it = np.loadtxt('CGM_prediction_data/insulin_therapy_train.csv', delimiter=',')

    # Load test data
    g_ = np.loadtxt('CGM_prediction_data/glucose_readings_test.csv', delimiter=',')
    c_ = np.loadtxt('CGM_prediction_data/meals_carbs_test.csv', delimiter=',')
    it_ = np.loadtxt('CGM_prediction_data/insulin_therapy_test.csv', delimiter=',')

    # Time align train & test data
    aligned_train_data = downscale(np.array([(g[i,:], c[i,:], it[i,:]) for i in range(g.shape[0])]), resolution)
    aligned_test_data = downscale(np.array([(g_[i,:], c_[i,:], it_[i,:]) for i in range(g_.shape[0])]), resolution)
    print(aligned_train_data.shape)

    # Break time aligned data into train & test samples
    if batch:
        train_data, train_label = process_data(aligned_train_data, time_horizon, ph)
        test_data, test_label = process_data(aligned_test_data, time_horizon, ph)

        return np.swapaxes(train_data,1,2), train_label, np.swapaxes(test_data,1,2), test_label

    else:

        return aligned_train_data, aligned_test_data


def load_uva(time_horizon, ph, resolution, batch):

    data = loadmat('uva-padova-data/sim_results.mat')
    train_data = np.zeros((231,3,1440))
    test_data = np.zeros((99,3,1440))

    # Separate train and test sets.. last 3 records of each patient will be used for testing
    count_train = 0
    count_test = 0
    for i in range(33):
        for j in range(10):

            if j>=7:
                test_data[count_test,0,:] = np.asarray(data['data']['results']['sensor'][count_test+count_train]['signals']['values']).flatten()[:1440]
                test_data[count_test,1,:] = np.asarray(data['data']['results']['CHO'][count_test+count_train]['signals']['values']).flatten()[:1440]
                test_data[count_test,2,:] = np.asarray(data['data']['results']['BOLUS'][count_test+count_train]['signals']['values']).flatten()[:1440] + np.asarray(data['data']['results']['BASAL'][i]['signals']['values']).flatten()[:1440]
                count_test+=1

            else:

                train_data[count_train,0,:] = np.asarray(data['data']['results']['sensor'][count_test+count_train]['signals']['values']).flatten()[:1440]
                train_data[count_train,1,:] = np.asarray(data['data']['results']['CHO'][count_test+count_train]['signals']['values']).flatten()[:1440]
                train_data[count_train,2,:] = np.asarray(data['data']['results']['BOLUS'][count_test+count_train]['signals']['values']).flatten()[:1440] + np.asarray(data['data']['results']['BASAL'][i]['signals']['values']).flatten()[:1440]
                count_train+=1

    train_data = downscale(train_data, resolution)
    test_data = downscale(test_data, resolution)

    if batch: 
        train_data, train_label = process_data(train_data, time_horizon, ph)
        test_data, test_label = process_data(test_data, time_horizon, ph)
    
        return np.swapaxes(train_data,1,2)*0.0555, train_label*0.0555, np.swapaxes(test_data,1,2)*0.0555, test_label*0.0555 # convert to mmol/L

    else:
        
        return train_data, test_data

# Custom metrics to examine performance of each element in the prediction horizon

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

if args.model=='lstm':

    print('Training lstm....')

    def lstm(ph, training):

        inp = tf.keras.Input(shape=(train_data.shape[1], train_data.shape[2]))
        model = tf.keras.layers.LSTM(200, return_sequences=True)(inp)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.LSTM(200, return_sequences=True)(model)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.LSTM(200, return_sequences=True)(model)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.Flatten()(model)
        model = tf.keras.layers.Dense(ph, activation=None)(model)

        x = tf.keras.Model(inputs=inp, outputs=model)

        x.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(), loss_metric1, loss_metric2, loss_metric3, loss_metric4, loss_metric5, loss_metric6])
        
        return x


    # Training LSTM

    train_data, train_label, test_data, test_label = load_uva(args.time_horizon, args.prediction_horizon, args.resolution, args.batch)

    print(train_data.shape, test_data.shape)

    model = lstm(args.prediction_horizon, args.training)

    print(model.summary())

    lstm = model.fit(x=train_data,
                     y=train_label,
                     batch_size=args.batch_size,
                     epochs=args.epochs,
                     validation_data=(test_data, test_label))

    model.save('saved/uva_padova_lstmv2.h5')
    json.dump(lstm.history, open('saved/uva_padova_lstm_historyv2', 'w'))


elif args.model=='crnn':

    print('Training CRNN....')

    def crnn(ph, training):
  
        inp = tf.keras.Input(shape=(train_data.shape[1], train_data.shape[2]))
        model = tf.keras.layers.Conv1D(256, 4, activation='relu', padding='same')(inp)
        model = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding='same')(model)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.Conv1D(512, 4, activation='relu', padding='same')(model)
        model = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding='same')(model)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.LSTM(200, return_sequences=True)(model)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.Flatten()(model)
        model = tf.keras.layers.Dense(ph, activation=None)(model)

        x = tf.keras.Model(inputs=inp, outputs=model)

        x.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(), loss_metric1, loss_metric2, loss_metric3, loss_metric4, loss_metric5, loss_metric6])
        
        return x


    train_data, train_label, test_data, test_label = load_uva(args.time_horizon, args.prediction_horizon, args.resolution, args.batch)

    print(train_data.shape, test_data.shape)

    model = crnn(args.prediction_horizon, args.training)

    print(model.summary())

    crnn = model.fit(x=train_data,
                     y=train_label,
                     batch_size=args.batch_size,
                     epochs=args.epochs,
                     validation_data=(test_data, test_label))

    model.save('saved/uva_padova_crnnv2.h5')
    json.dump(crnn.history, open('saved/uva_padova_crnn_historyv2', 'w'))


elif args.model=='bilstm':

    print('Training bidirectional LSTM....')

    def bilstm(ph, training):

        inp = tf.keras.Input(shape=(train_data.shape[1], train_data.shape[2]))
        model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True))(inp)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True))(model)
        model = tf.keras.layers.Dropout(rate=0.5)(model, training=training)
        model = tf.keras.layers.Flatten()(model)
        model = tf.keras.layers.Dense(ph, activation=None)(model)

        x = tf.keras.Model(inputs=inp, outputs=model)

        x.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError(), loss_metric1, loss_metric2, loss_metric3, loss_metric4, loss_metric5, loss_metric6])
        
        return x 


    train_data, train_label, test_data, test_label = load_uva(args.time_horizon, args.prediction_horizon, args.resolution, args.batch)

    print(train_data.shape, test_data.shape)

    model = bilstm(args.prediction_horizon, args.training)

    print(model.summary())

    bilstm = model.fit(x=train_data,
                     y=train_label,
                     batch_size=args.batch_size,
                     epochs=args.epochs,
                     validation_data=(test_data, test_label))

    model.save('saved/uva_padova_bilstmv2.h5')
    json.dump(bilstm.history, open('saved/uva_padova_bilstm_historyv2', 'w'))

else:

    print('Set args.model to lstm, bilstm, or crnn to train the respective architecture.')

