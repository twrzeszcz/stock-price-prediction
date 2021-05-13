import streamlit as st
import pandas as pd
import pandas_datareader as pdr
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import datetime
import time
import sklearn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import gc
st.set_option('deprecation.showPyplotGlobalUse', False)

gc.enable()

np.random.seed(120)
tf.random.set_seed(120)


def load_data(company):
    df = pdr.DataReader(company, 'yahoo')
    return df


def preprocess(df_close):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_close_scaled = scaler.fit_transform(df_close.values.reshape(-1, 1))

    num_steps = 100
    X = [df_close_scaled[i:i+num_steps] for i in range(len(df_close_scaled) - (num_steps + 10))]
    Y = np.empty((len(X), 100, 10))
    for j in range(len(X)):
        for i in range(num_steps):
            Y[j, i, :] = df_close_scaled[i + j:j + i + 10, 0]

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    batch_size = 32
    dataset = dataset.shuffle(10000).batch(batch_size)

    train_size = int(0.8 * len(dataset))
    train_set = dataset.take(train_size)
    val_set = dataset.skip(train_size)

    del X, Y, dataset

    return train_set, val_set, df_close_scaled, scaler


def train(train_set, val_set):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for rate in (1, 2, 4, 8) * 2:
        model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal",
                                      activation="relu", dilation_rate=rate))
    model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])

    t1 = time.time()
    model.fit(train_set, epochs=100, validation_data=val_set)
    t2 = round(time.time() - t1, 2)

    del train_set, val_set
    st.success('Successfully trained')
    st.info('Training time: {}'.format(t2) + 's')

    return model



def predict(scaler, model, df_close_scaled, df_close, num_steps):
    for step in range(int(num_steps / 10)):
        pred = model.predict(np.expand_dims(df_close_scaled, axis=0))[:, -1, :]
        df_close_scaled = np.concatenate([df_close_scaled, pred.T], axis=0)

    if df_close.index[-1] == datetime.date.today():
        df_pred = pd.DataFrame(scaler.inverse_transform(df_close_scaled[-num_steps:]), columns=['Close'],
                               index=pd.date_range(datetime.date.today() + datetime.timedelta(days=1), freq='D', periods=num_steps))
    else:
        df_pred = pd.DataFrame(scaler.inverse_transform(df_close_scaled[-num_steps:]), columns=['Close'],
                               index=pd.date_range(datetime.date.today(), freq='D', periods=num_steps))

    df_close_all = pd.concat([df_close, df_pred], axis=0)
    df_close_all.rename(columns={0: 'Original', 'Close': 'Predicted'}, inplace=True)

    del pred, df_pred

    return df_close_all

def visualize(df_close_all):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_close_all.index, y=df_close_all['Original'],
                             mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=df_close_all.index, y=df_close_all['Predicted'],
                             mode='lines+markers', name='Predicted'))
    fig.update_layout(title='Original data + predicted values', title_x=0.5, xaxis_title='Date', yaxis_title='Closing Price')
    st.write(fig)

def main_section():
    st.title('Stock Price Prediction Project')
    background_im = cv2.imread('images/stock.jpeg')
    st.image(cv2.cvtColor(background_im, cv2.COLOR_BGR2RGB), use_column_width=True)
    st.subheader('General info')
    st.info('This app serves a purpose of predicting the next days stock prices for different companies chosen by the user.')
    st.info('This is done in the Live Prediction section where the user specifies a company and then the data is '
            'fetched from the https://finance.yahoo.com/.')
    st.info('After preprocessing it is fed into the model which is trained in real time. User can choose for how many days prices should be predicted.'
            ' We use here the simplified version of the Wavenet model shown in the jupyter notebook on the github.')
    del background_im
    gc.collect()

def live_prediction():
    st.title('Live Prediction')

    companies = {
        'Amazon': 'AMZN',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Tesla': 'TSLA',
        'Sony': 'SONY',
        'Boeing': 'BA',
        'McDonalds': 'MCD',
        'IBM': 'IBM',
        'Pepsi': 'PEP',
        'Cocal Cola': 'KO',
        'Intel': 'INTC',
        'AMD': 'AMD'
    }
    selected_company = st.sidebar.selectbox('Select company', list(companies.keys()))
    if st.sidebar.checkbox('Fetch the data'):
        df = load_data(companies[selected_company])
        df_close = df['Close']
        st.success('Data successfully loaded')
        st.dataframe(df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_close.index, y=df_close.values, mode='lines'))
        fig.update_layout(title='Original data - Closing Prices', title_x=0.5, xaxis_title='Date', yaxis_title='Closing Price')
        st.write(fig)

        num_steps = st.sidebar.selectbox('Select how many time steps to predict', list(range(10, 101, 10)))
        if st.sidebar.button('Predict closing prices for the next {} days'.format(num_steps)):
            train_set, val_set, df_close_scaled, scaler = preprocess(df_close)

            model = train(train_set, val_set)
            df_close_all = predict(scaler, model, df_close_scaled, df_close, num_steps)
            visualize(df_close_all)

            del train_set, val_set, df_close_scaled, scaler, df_close_all, model
            gc.collect()



activities = ['Main', 'Live Prediction', 'About']
option = st.sidebar.selectbox('Select Option', activities)

if option == 'Main':
    main_section()

if option == 'Live Prediction':
    live_prediction()
    gc.collect()

if option == 'About':
    st.title('About')
    st.write('This is an interactive website for the Stock Price Prediction Project. Data was taken from https://finance.yahoo.com/.')
