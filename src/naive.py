#该代码主要是应用一些比较简单的模型对数据进行预测分析
import pandas as pd
import numpy as np
from data_pre_mulvarible import *
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
    # 先读取数据
    data = pd.read_csv("../2015-2017beijing.csv")
    # 对缺失值进行处理,先对缺失值进行差分处理
    data = data.interpolate()
    # 在对缺失值进行向下填充
    data.fillna(method='bfill', inplace=True)
    data_ = data[['date', 'pm2.5']]
    data_frame = pd.DataFrame({})
    for i in range(3):
        print(i+1)
        data_['pred'] = data_['pm2.5'].shift(int(i+1))
        data_['date'] = pd.to_datetime(data_['date'])
        data_.index = data_['date']
        data_test = data_['2017-06':'2017-12']
        y_pred = data_test['pred'].values
        y_true = data_test['pm2.5'].values
        data_frame['real value'] = y_true.reshape(1, -1)[0]
        data_frame['horizon=%d'%(i+1)] = y_pred.reshape(1, -1)[0]
        print("rmse为:", np.sqrt(test_mse(y_true, y_pred)))
        print("mape为:", test_mape(y_true, y_pred))
        print("mae为:", test_mae(y_true, y_pred))
    data_frame.to_excel("naive.xlsx", index=False)