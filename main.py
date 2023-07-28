from upbit import Upbit
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
# import pyupbit as 


def get_rsi_data():
    period = 14
    upbit_b = Upbit()
    data = upbit_b.fetch_as_df("KRW-ETH")
    print(data)
    data = data.sort_index(ascending=False)
    delta = data['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(com=(period-1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period-1), min_periods=period).mean()
    RS = _gain / _loss

    rsi_14 = pd.Series(100 - (100 / (1 + RS)), name = "rsi")
    data['rsi'] = rsi_14
    data = data.sort_index(ascending=True)
    data = data[0:3360]

    # data.set_index('time', inplace=True)
    # df_data=data.astype(float)
    return data

def get_rsi_avg_data():
    period = 14
    upbit_b = Upbit()
    data = upbit_b.fetch_as_df("KRW-ETH")
    data = data.sort_index(ascending=False)
    delta = data['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(com=(period-1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period-1), min_periods=period).mean()
    RS = _gain / _loss

    rsi_14 = pd.Series(100 - (100 / (1 + RS)), name = "rsi")
    data['rsi'] = rsi_14
    data = data.sort_index(ascending=True)
    data = data[0:3360]
    print(data)
    time_for_data = list()
    result_rsi = 0
    # data_for_matplotlib(data, 'time', 'rsi')

    for i in range(0, 24):
        time_format = data.iloc[i]['time'].split("T")[1].split(":")[0]
        # each_data_for_time = list()
        # 시간별로의 rsi 평균 계산하기
        time_for_rsi_value = 0
        count = 0 
        for j in range(i, len(data), 24):
            if data.iloc[j]['time'].split("T")[1].split(":")[0] == time_format:
                time_for_rsi_value += data.iloc[j]['rsi']
                count += 1
                result_rsi = time_for_rsi_value / count
            else:
                pass
            
        time_for_data.append([time_format, result_rsi])   
    time_for_data_df = pd.DataFrame(time_for_data, columns=['time', 'rsi'])
    time_for_data_df = time_for_data_df.sort_values(by=['time'])
    print(time_for_data_df)
    data_for_matplotlib(time_for_data_df, 'time', 'rsi')
    return time_for_data_df


# y축은 종가, x 축은 24시간이기 대문에 24시간에 대한 시간 


# 10일치 데이터의 평균 
def get_min_avg_data():
    # time_for_data = [[time, sum_data], [time, sum_data]]
    time_for_data = list()

    upbit_b = Upbit()
    df_data = upbit_b.fetch_as_df("KRW-ETH")
    print(df_data)
    for i in range(0, 24):

        time_format = df_data.iloc[i]['time'].split("T")[1].split(":")[0]
        # each_data_for_time = list()
        # 시간별로의 종가 계산 
        time_for_close_value = 0
        count = 0 
        for j in range(i, len(df_data), 24):
            if df_data.iloc[j]['time'].split("T")[1].split(":")[0] == time_format:
                time_for_close_value += df_data.iloc[j]['close']
                count += 1
                time_for_close_value = time_for_close_value / count
            else:
                pass
        
        time_for_data.append([time_format, time_for_close_value])   
    
    time_for_data_df = pd.DataFrame(time_for_data, columns=['time', 'rsi'])
    time_for_data_df = time_for_data_df.sort_values(by=['time'])
    
    # data_for_matplotlib(time_for_data_df, 'time', 'close')
    return time_for_data_df

def get_sklearn_model():
    data = get_rsi_avg_data()
    # for i in range(0, len(data)):
    #     time = data.iloc[i]['time'].split("T")[1].split(":")[0]
    #     data.iloc[i]['time'].replace(data.iloc[i]['time'], time)
    # #data.iloc[i]['time'].split("T")[1].split(":")[0]
    # for i in range(0, len(data)):
    #     data.iloc[i]['time'].replace(data.iloc[i]['time'], "i")

    # print(data)
    # time_for_data_df = pd.DataFrame(data, columns=['time', 'close'])
    # time_for_data_df = time_for_data_df.sort_values(by=['time'])
    data=data.astype(float)

    Y = data['rsi']
    X = data.drop(['rsi'], axis = 1, inplace = False)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 156)
    lr = LinearRegression() # 선형 회귀 분석 모델 객체 lr 생성
    lr.fit(X_train, Y_train) # training 데이터로 학습 수행
    Y_predict = lr.predict(X_test)
    print(Y_predict)
    mse = mean_squared_error(Y_test, Y_predict)
    rmse = np.sqrt(mse)
    print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse)) 
    print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))

    print('Y 절편 값: ', lr.intercept_)
    print('회귀 계수 값: ', np.round(lr.coef_, 1))
    coef = pd.Series(data = np.round(lr.coef_, 2), index = X.columns) 
    coef.sort_values(ascending = False)

    fig, ax = plt.subplots(figsize = (16, 16), ncols = 1, nrows = 1)
    print(fig, ax)
    x_features =['rsi']
    sns.regplot(x = x_features, y = 'rsi', data = data, ax = axs[row][col])
    # for i, feature in enumerate(x_features):
    #     row = int(i/3)
    #     col = i%3
    #     sns.regplot(x = feature, y = 'rsi', data = data, ax = axs[row][col])

def data_for_matplotlib(data, x_name, y_name):
    plt.figure(figsize = (12, 5))
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid(True)
    plt.plot(range(len(data)), data[y_name]) 
    plt.xticks(range(len(data)), data[x_name]) 
    # data.plot(kind='bar', x='Time', y='Close')
    plt.show()


def all_actions():
    # close_data_df = get_min_data()
    # data_for_matplotlib(close_data_df)
    # get_min_data()
    # get_rsi_data()
    # get_min_avg_data()
    get_sklearn_model()

if __name__ == "__main__":
    all_actions()


