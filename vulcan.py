# __author__=Ryan
# created_date = 20180213
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
import sys
os.chdir(r'E:\gridsum\test\rnn')


# 转化成相应的输入输出矩阵
def create_data_set(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


if __name__ == '__main__':
    np.random.seed(7)

    df = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)

    data_set = df.values
    data_set = data_set.astype('float32')
    # plt.plot(data_set)
    # plt.show()

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_set = scaler.fit_transform(data_set)
    train_size = int(len(data_set)*0.7)
    test_size = len(data_set) - train_size
    train, test = data_set[0: train_size, :], data_set[train_size: len(data_set), :]

    look_back = 12
    train_x, train_y = create_data_set(train, look_back)
    test_x, test_y = create_data_set(test, look_back)

    # lstm输入格式[samples, time_steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    # train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    # test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    # 建立lstm
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    # model.add(LSTM(4, input_shape=(look_back, 1)))
    # model.add(LSTM(4, input_shape=(1, look_back), activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_x, train_y, batch_size=1, verbose=2, epochs=100)

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])

    train_score = math.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    print('训练误差', train_score)
    test_score = math.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print('测试误差', test_score)

    train_predict_plot = np.empty_like(data_set)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back: len(train_predict)+look_back, :] = train_predict

    test_predict_plot = np.empty_like(data_set)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict)+(look_back*2)+1:len(data_set)-1, :] = test_predict

    plt.plot(scaler.inverse_transform(data_set))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.legend(['real', 'train', 'test'])
    plt.show()







