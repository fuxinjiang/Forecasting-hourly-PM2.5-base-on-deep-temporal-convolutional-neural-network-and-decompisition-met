
# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
from sklearn.svm import SVR
import math
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as pyplot
import numpy as np
from random import shuffle
import _pickle as pickle
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import prettytable
import random
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
def linear_regression(Train_Data, Train_label):
    reg = LinearRegression().fit(Train_Data, Train_label)
    #print("回归R^2为:", reg.score(Train_Data, Train_label))
    #print("回归截距为:", reg.intercept_)
    #print("回归系数为:", reg.coef_)
    return reg
def data_main(df, chosen_scale, lag, horizon):
    data = df.copy()
    #首先确定测试集的长度
    data.index = data['date']
    test_length = len(data['2017-06':'2017-12'])
    data.reset_index(drop = True, inplace = True)
    x_con = np.array(data[['pm2.5', 'pm10', 'so2', 'no2', 'o3', 'co', '温度', '湿度', '降雨量', 'wind_x', 'wind_y','大气压', 'hour', 'dayofweek', 'month', 'Imf1','Imf2','Imf3','Imf4','Imf5','Imf6','Imf7','Imf8','Imf9','Imf10','Imf11', 'Imf12', 'Imf13', 'Imf14','Imf15', 'Residual']])
    #先用训练集的数据计算标准化的指标，然后对整个数据集进行转换
    x_con_train = x_con[:-test_length,:]
    print(x_con_train.shape)
    if chosen_scale == "NormalizedScaler":
        scale = NormalizedScaler(min=np.min(x_con_train, axis=0), max=np.max(x_con_train, axis=0))
    elif chosen_scale == "StandardScaler":
        scale = StandardScaler(mean=np.mean(x_con_train, axis=0), std=np.std(x_con_train, axis=0))
    else:
        print("please chosen a scale between NormalizedScaler and StandardScaler")
        return None
    x_con_nor = scale.transform(x_con)
    length = len(data) - lag - horizon + 1
    x_matrix = np.zeros((length, lag * 31))
    y_matrix = np.zeros((length, 1))
    for i in range(length):
        x_matrix[i, :] = x_con_nor[i:i + lag, :].reshape((1,-1))
        # 这个地方要注意周期的问题
        y_matrix[i, :] = x_con_nor[i + lag + horizon - 1, 0]
    return scale, x_matrix, y_matrix, test_length
#对数据进行标准化
class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    #主要用于后面进行单变量的损失函数操作
    def inverse_transform(self, data):
        return data * self.std[0] + self.mean[0]
#进行归一化
class NormalizedScaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    def inverse_transform(self, data):
        return data * (self.max[0] - self.min[0]) + self.min[0]
def SupportVectorRegression(C, gamma, Train_X, Train_Y):
    lin_clf = SVR(kernel='rbf', epsilon=0.01, C=C, gamma=gamma, n_jobs=-1)
    #shuffle=True 用于每次交叉验证，数据集划分的结果不同
    lin_clf.fit(Train_X, Train_Y)
    return lin_clf

def KoldValidation_SupportVectorMachine(C_range, gamma_range, K, Train_Data, Train_label):
    param_grid = dict(gamma=gamma_range, C=C_range)
    #print(param_grid)
    lin_clf = SVR(kernel='rbf')
    #shuffle=True 用于每次交叉验证，数据集划分的结果不同
    KF = KFold(n_splits=K, shuffle=True)
    gsearch1 = GridSearchCV(estimator = lin_clf, param_grid=param_grid, scoring='neg_mean_squared_error', cv=KF, n_jobs=-1)
    gsearch1.fit(Train_Data, Train_label)
    #print('搜寻过程', gsearch1.cv_results_)
    print('最佳参数组合', gsearch1.best_params_)
    print('最优', gsearch1.best_score_ )
    return gsearch1


def MLP(hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, max_iter, Train_X, Train_Y):
    '''
    hidden_layer_sizes，隐藏层结构，一般为一个tuple
	activation 为激活函数， {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
	solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’优化方法
	alpha : float, optional, default 0.0001 L2 penalty (regularization term) parameter. 正则化系数
	batch_size : int, optional, default ‘auto’
	Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)
	learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
	'''
    mlp_ = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver= solver, alpha = alpha, batch_size = batch_size, learning_rate = learning_rate, learning_rate_init = learning_rate_init, max_iter=max_iter)
    mlp_.fit(Train_X, Train_Y)
    return mlp_

def KoldValidation_Mlp(hs, bs, max_iter, K, Train_Data, Train_label):
    '''
    hidden_layer_sizes，隐藏层结构，一般为一个tuple
    activation 为激活函数， {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
    solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’优化方法
    alpha : float, optional, default 0.0001 L2 penalty (regularization term) parameter. 正则化系数
    batch_size : int, optional, default ‘auto’
    Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)
    learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
    '''
    param_grid = dict(hidden_layer_sizes=hs, batch_size=bs, max_iter=max_iter)
    mlp_ = MLPRegressor(activation='identity', solver='adam', alpha=0.1, learning_rate_init=0.01)
    KF = KFold(n_splits=K, shuffle=True)
    gsearch1 = GridSearchCV(estimator = mlp_, param_grid=param_grid, scoring='neg_mean_squared_error', cv=KF, n_jobs=-1)
    gsearch1.fit(Train_Data, Train_label)
    print('最佳参数组合', gsearch1.best_params_)
    print('最优', gsearch1.best_score_)
    return gsearch1


#线性回归模型

def test_mse(y_true, y_pred):
    mse = np.square(y_true - y_pred)
    return np.mean(mse)
def test_mae(y_true, y_pred):
    mae = np.abs(y_true - y_pred)
    return np.mean(mae)
def test_mape(y_true, y_pred):
    mape = np.abs(y_true - y_pred) / np.abs(y_true)
    return np.mean(mape)
def test_rmse(y_true, y_pred):
    rmse = np.square(y_true - y_pred)
    return np.sqrt(np.mean(rmse))
if __name__ == "__main__":
    data = pd.read_csv("../2015-2017beijing.csv")
    # 对缺失值进行处理,先对缺失值进行差分处理
    data = data.interpolate()
    # 在对缺失值进行向下填充
    data.fillna(method='bfill', inplace=True)
    weather_data = pd.read_csv("../气候因素/天气数据_小时.csv")
    weather_data.rename(columns={"日期": 'date', '小时': 'hour'}, inplace=True)
    data = pd.merge(data, weather_data, how='left', on=['date', 'hour'])
    data['date'] = pd.to_datetime(data['date'])
    data['dayofweek'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month - 1
    data['wind_x'] = data['风速']* np.cos(data['风向角度'] / 180*math.pi)
    data['wind_y'] = data['风速'] * np.sin(data['风向角度'] / 180 * math.pi)
    ceemdan_data = pd.read_excel("分解数据.xlsx")
    data = pd.merge(data, ceemdan_data, how='left', on=['date', 'hour'])
    lag = 24
    for horizon in range(1,4):
        #test
        scale, x_matrix, y_matrix, test_length = data_main(data, 'StandardScaler', lag, horizon)
        x_matrix_train = x_matrix[:-test_length, :]
        x_matrix_test = x_matrix[-test_length:, :]
        y_matrix_train = y_matrix[:-test_length, :]
        y_matrix_test = y_matrix[-test_length:, :]

        #计算线性回归的
        print("开始线性回归的计算")
        reg1 = linear_regression(x_matrix_train, y_matrix_train)
        reg_pred_y = reg1.predict(x_matrix_test).reshape((-1, 1))
        reg_pred_y = scale.inverse_transform(reg_pred_y).reshape((-1, 1))
        reg_true_y = scale.inverse_transform(y_matrix_test)
        data_frame = pd.DataFrame({})
        data_frame['real value'] = reg_true_y.reshape(1, -1)[0]
        data_frame['pred value'] = reg_pred_y.reshape(1, -1)[0]
        data_frame = pd.DataFrame(data_frame)
        data_frame.to_excel("ceemdan_lr_horizon_%d.xlsx" %horizon, index=False)
        # 开始计算误差
        print(test_mse(reg_true_y, reg_pred_y))
        print(test_mae(reg_true_y, reg_pred_y))
        print(test_mape(reg_true_y, reg_pred_y))

    
    #计算bpnn的
    '''
    train_list = list(range(x_matrix_train.shape[0]))
    shuffle(train_list)
    x_train = x_matrix_train[train_list, :]
    y_train = y_matrix_train[train_list, :]
    
    mlp = MLP(hidden_layer_sizes=(16,16,), activation='relu', solver='adam', alpha=0, batch_size=256, learning_rate='constant', learning_rate_init=0.01, max_iter=1000000, Train_X = x_train, Train_Y = y_train)
    test_y = mlp.predict(x_matrix_test)
    test_y = scale.inverse_transform(test_y).reshape((-1, 1))
    test_y_true = scale.inverse_transform(y_matrix_test).reshape((-1, 1))
    print(test_mse(test_y_true, test_y))
    print(test_mae(test_y_true, test_y))
    print(test_mape(test_y_true, test_y))
    '''
    '''
    print("开始bpnn的计算")
    train_list = list(range(x_matrix_train.shape[0]))
    shuffle(train_list)
    x_train = x_matrix_train[train_list, :]
    y_train = y_matrix_train[train_list, :]
    K=5
    hidden_size = np.array([64,128])
    batch_size = np.array([8, 16, 32, 64, 128, 256])
    max_iter = np.array([100, 300, 500])
    #mlp = KoldValidation_Mlp(hidden_size, batch_size, max_iter, K, x_train, y_train.reshape(-1,))
    
    mlp = MLP(hidden_layer_sizes=(32,16,), activation='relu', solver='adam', alpha=0, batch_size=128, learning_rate='constant', learning_rate_init=0.01, max_iter=300, Train_X = x_train, Train_Y = y_train)
    mlp_pred_y = mlp.predict(x_matrix_test)
    mlp_pred_y = scale.inverse_transform(mlp_pred_y).reshape((-1, 1))
    mlp_true_y = scale.inverse_transform(y_matrix_test)
    data_frame = pd.DataFrame({})
    data_frame['real value'] = mlp_true_y.reshape(1, -1)[0]
    data_frame['pred value'] = mlp_pred_y.reshape(1, -1)[0]
    data_frame = pd.DataFrame(data_frame)
    data_frame.to_excel("mlp_horizon_%d.xlsx" %horizon, index=False)
    # 开始计算误差
    print(test_mse(mlp_true_y, mlp_pred_y))
    print(test_mae(mlp_true_y, mlp_pred_y))
    print(test_mape(mlp_true_y, mlp_pred_y))
    '''
    #计算svr的
    '''
    K = 5
    C_range = np.logspace(-3, 3, 11)
    gamma_range = np.logspace(-2, 2, 11)
    SVM_ = KoldValidation_SupportVectorMachine(C_range, gamma_range, K, x_train, y_train.reshape(-1,)) 
    svm_pred_y = SVM_.predict(x_matrix_test)
    svm_pred_y = scale.inverse_transform(svm_pred_y).reshape((-1, 1))
    svm_true_y = scale.inverse_transform(y_matrix_test)
    data_frame = pd.DataFrame({})
    data_frame['real value'] = svm_true_y.reshape(1, -1)[0]
    data_frame['pred value'] = svm_pred_y.reshape(1, -1)[0]
    data_frame = pd.DataFrame(data_frame)
    data_frame.to_excel("svm_horizon_%d.xlsx" %horizon, index=False)
    # 开始计算误差
    print(test_mse(svm_true_y, svm_pred_y))
    print(test_mae(svm_true_y, svm_pred_y))
    print(test_mape(svm_true_y, svm_pred_y))
    '''