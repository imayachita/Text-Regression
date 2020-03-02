import pandas as pd
import numpy as np

import os

import tensorflow as tf
from tensorflow import keras

import json
import utils

import time
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt
import seaborn as sns


tf.compat.v1.set_random_seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To block TensorFlow warning


class TextRegModel:
    def __init__(self,config_file):
        if config_file is not None:
            with open(config_file) as f:
                config = json.load(f)

        self.model_type = config['model_type']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.stopwords_file = config['stopwords_file']
        self.max_len = config['maxlen']
        self.train = config['train_file']
        self.model_dir = config['model_dir']
        self.embedding_model = config['word_embedding']
        self.limit = config["max_number_of_examples_per_class"]
        self.input_col = config['input_column']
        self.target_col = config['target_column']
        self.predict_data = config['predict_file']
        self.eval_data = config['eval_file']


        print('model type: ', self.model_type)
        assert self.model_type == "BILSTM" or self.model_type == "CNN", "Model type must be BILSTM or CNN."

        #no limit for examples of each class
        if self.limit == "None":
            self.limit = None


    def preprocess(self,train,test,mode):
        '''
        Preprocess data and do feature extraction
        Args:
            train (dataframe): train data in dataframe format
            test (dataframe): validation data in dataframe format
            mode (str): mode whether it is "training", "eval", "prediction"
        Returns:
            X: train data with shape=[num_examples,max_len,embedding_dim]
            Y: train label with shape=[num_examples,]
            test_X: validation data with shape=[num_examples,max_len,embedding_dim]
            test_Y: validation label with shape=[num_examples,]
        '''
        embeddings_dict,embedding_dim = utils.read_embedding(self.embedding_model)
        num_words = len(embeddings_dict)
        X,Y,test_X,test_Y= utils.create_embedding_features(
                                                            mode,
                                                            train,
                                                            test,
                                                            self.input_col,
                                                            self.target_col,
                                                            embeddings_dict,
                                                            embedding_dim,
                                                            stopwords_file=self.stopwords_file,
                                                            max_len=self.max_len,
                                                            limit=self.limit
                                                            )
        return X,Y,test_X,test_Y,num_words,embedding_dim


    def training(self):
        '''
        Train model
        '''

        assert self.train != "None", "Please provide training dataset"

        model_dir = self.model_dir
        model_type = self.model_type
        input_col = self.input_col
        target_col = self.target_col

        #read dataset
        train = pd.read_csv(self.train)

        x = train[input_col]
        y = train[target_col]

        #split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

        train = pd.concat([X_train,y_train],axis=1)
        test = pd.concat([X_test,y_test],axis=1)

        #create model directory
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        #preprocess data
        self.X,self.Y,self.test_X,self.test_Y,num_words,embedding_dim = self.preprocess(train,test,"training")

        X = self.X
        Y = self.Y
        test_X = self.test_X
        test_Y = self.test_Y

        print('shape: ', X.shape, Y.shape, test_X.shape, test_Y.shape)

        #define model
        if model_type == "BILSTM":
            model = BILSTM(X,mc=True)
        elif model_type == "CNN":
            model = CNN(X,mc=True)


        print('Start Training...')


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                      loss='mean_squared_error',
                      metrics=['mse','mae']
                     )
        print(model.summary())

        tbCallBack = tf.keras.callbacks.TensorBoard(
                                    log_dir=model_dir,
                                    histogram_freq=0,
                                    write_graph=True,
                                    write_images=True
                                    )

        # es = tf.keras.callbacks.EarlyStopping(monitor='loss')

        model.fit(
                    X, Y,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1,
                    validation_data=(test_X, test_Y),
                    callbacks=[tbCallBack]
                    )

        #saving model
        tf.keras.models.save_model(model,os.path.join(model_dir,'model.h5'))

        print('Done Training...')
        self.evaluate(mode='training')

    def measure_uncertainty(self,model_mc,X_test,Y_test):
        '''
        Measuring uncertainty by comparing the prediction from Monte Carlo Dropout
        Args:
            model_mc: model with Monte Carlo Dropout
            X_test: test data
            Y_test: test label
        '''
        model_dir = self.model_dir
        pred_mc = []
        sample_size = 50
        print('Do Prediction with Monte Carlo Dropout model')
        for t in range(sample_size):
            pred_mc.append(model_mc.predict(X_test, batch_size=1000))
        pred_mc_array = np.array(pred_mc)

        print('Calculating mean and standard deviation')
        pred_mc_mean = pred_mc_array.mean(axis=0)
        pred_mc_std = pred_mc_array.std(axis=0)

        accs = []
        for p in pred_mc:
            accs.append(mean_squared_error(Y_test,p))
        print('Monte Carlo MSE: ', sum(accs)/len(accs))

        ensemble_acc = mean_squared_error(Y_test,pred_mc_mean.reshape(-1,))
        print('Monte Carlo Ensemble MSE: ', ensemble_acc)

        return pred_mc_mean.reshape(-1,), pred_mc_std.reshape(-1,)


    def evaluate(self,mode=None):
        '''
        Evaluate model
        '''

        print('Start Evaluating...')
        model_dir = self.model_dir

        if mode=='training': #do evaluation on stratified 20% of training data
            test_X = self.test_X
            test_Y = self.test_Y
            filename = 'Evaluation_Report_on_Training.txt'

        else: #do evaluation on eval_file provided in JSON config file
            assert self.eval_data != "None", "Please provide evaluation dataset"
            eval_data = self.eval_data
            eval_data = pd.read_csv(eval_data)
            test_X, test_Y, _, _, _, _ = self.preprocess(eval_data,None,"evaluation")
            filename = 'Evaluation_Report_on_Eval_Data.txt'


        #load model
        model = tf.keras.models.load_model(os.path.join(model_dir,'model.h5'))

        #do evaluation
        eval_res = model.evaluate(test_X,test_Y)
        print('Evaluation results [Loss, MSE, MAE]: ', eval_res)

        #save evaluation results
        with open(os.path.join(model_dir,filename), 'w') as f:
            f.write('** Evaluation Report **\n')
            f.write('Loss: ' + str(eval_res[0]) + '\n')
            f.write('Mean Squared Error: ' + str(eval_res[1]) + '\n')
            f.write('Mean Absolute Error: ' + str(eval_res[2]) + '\n')

        if mode!="training":
            pred = model.predict(test_X)
            eval_data['Prediction'] = pred
            print('Measuring Uncertainty..')
            pred_mc_mean,pred_mc_std = self.measure_uncertainty(model,test_X,test_Y)
            eval_data['Uncertainty_Mean'] = pred_mc_mean
            eval_data['Uncertainty_StandardDev'] = pred_mc_std
            print('Done Measuring Uncertainty.')
            eval_data.to_csv(os.path.join(model_dir,'Evaluation_Results.csv'))

        print('Done Evaluating...')


    def predict(self):
        '''
        Do prediction
        '''
        assert self.predict_data != "None", "Please provide dataset to predict"
        print('Start Prediction')
        model_dir = self.model_dir

        #load data
        predict_data = self.predict_data
        predict_data = pd.read_csv(predict_data)

        #do preprocess
        pred_X, _ , _, _, _, _ = self.preprocess(predict_data,None,"prediction")

        #load model
        model = tf.keras.models.load_model(os.path.join(model_dir,'model.h5'))

        #do prediction
        pred_result = model.predict(pred_X, verbose=2)

        #save results
        predict_data['Prediction'] = pred_result
        predict_data.to_csv(os.path.join(model_dir,'Prediction_Results.csv'))
        print('Prediction Done!')




#########################################
# LIST OF ALL MODELS

def dropout(X, p=0.5, mc=False):
    '''
    Implement Monte Carlo Dropout for Model Uncertainty
    Args:
        X: training data
        p: dropout rate
        mc: flag whether model is trained with dropout training True or not
    Returns:
        Tensorflow keras layers
    '''
    if mc:
        return tf.keras.layers.Dropout(p)(X, training=True)
    else:
        return tf.keras.layers.Dropout(p)(X)

def BILSTM(X,mc=False):
    '''
    Implementation of Bi-LSTM architecture on Tensorflow Keras
    Args:
        X: train data in np.array with shape=[num_examples,max_len,embedding_dim]
    Returns:
        model = model instance
    '''
    maxlen = X.shape[1]
    embedding_dim = X.shape[2]

    inputs = tf.keras.layers.Input(shape=(maxlen,embedding_dim))
    model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True))(inputs)
    model = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True))(model)
    model = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50, activation='relu'))(model)
    model = dropout(model, p=0.2, mc=mc)
    model = tf.keras.layers.Flatten()(model)
    model = dropout(model, p=0.2, mc=mc)
    model = tf.keras.layers.Dense(1)(model)

    model = tf.keras.Model(inputs=inputs,outputs=model)

    return model


def CNN(X,mc=False):
    '''
    Implementation of CNN architecture on Tensorflow Keras
    Args:
        X: train data in np.array with shape=[num_examples,max_len,embedding_dim]
    Returns:
        model = model instance
    '''

    maxlen = X.shape[1]
    embedding_dim = X.shape[2]

    inputs = tf.keras.layers.Input(shape=(maxlen,embedding_dim))
    model = tf.keras.layers.Conv1D(64,
              kernel_size = 5,
              strides = 2,
              kernel_initializer = 'glorot_normal',
              bias_initializer='glorot_normal',
              padding='same')(inputs)
    model = tf.keras.layers.Activation('relu')(model)

    model = tf.keras.layers.Conv1D(128,
              kernel_size = 3,
              strides = 1,
              padding='same')(model)
    model = tf.keras.layers.Activation('relu')(model)

    model = tf.keras.layers.Conv1D(200,
              kernel_size = 3,
              strides = 1,
              padding='same')(model)
    model = tf.keras.layers.Activation('relu')(model)
    model = tf.keras.layers.SpatialDropout1D(0.5)(model)
    model = tf.keras.layers.GlobalMaxPooling1D()(model)
    model = tf.keras.layers.Dense(150, activation='relu')(model)
    model = dropout(model, p=0.2, mc=mc)
    model = tf.keras.layers.Dense(50, activation='relu')(model)
    model = dropout(model, p=0.2, mc=mc)

    model = tf.keras.layers.Dense(1)(model)
    model = tf.keras.Model(inputs=inputs,outputs=model)

    return model
