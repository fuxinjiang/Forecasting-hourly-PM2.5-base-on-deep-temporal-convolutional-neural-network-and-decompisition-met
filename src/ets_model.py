# %%ets预测
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import ExponentialSmoothing
from tqdm import tqdm
def ParameterSelection(data, p, q):
    
    arma_model = sm.tsa.statespace.SARIMAX(data, order=(p,0,q), seasonal_order=(0,0,0,0))
    arma_model = arma_model.fit()
    #print('参数p和q分别选择为%d,%d时,模型AIC,BIC,HQIC值如下:' %(p,q))
    #print(60*'*')
    print(p, q, arma_model.aic,arma_model.bic,arma_model.hqic)
    #做D-W检验，德宾-沃森检验，是目前检验自相关性最常用的方法，但它只适用于检验一阶自相关性。因为自相关系数\rou值介于－1和1之间
    #所以0<=DW<=4。
    #当DW值显著的越接近于0或者4时，存在自相关性，而接近于2时，则不存在(一阶)自相关性。一般对残差进行自相关性检验
    #print('残差的Durbin-Watson检验结果如下:')
    #print(60*'*')
    #print(sm.stats.durbin_watson(arma_model.resid.values))
    #返回模型的残差
    return arma_model, arma_model.resid, arma_model.predict()

def ETS(data):
    ets = ExponentialSmoothing(data,
                         trend='add',
                         damped_trend=True,
                         seasonal='mul',
                         seasonal_periods=24,
                         dates=data.index.values,
                         freq='H',
                         initialization_method='estimated',
                         use_boxcox=True,
                         missing='drop').fit()
    return ets
    
    
def mse(true, predict):
    return np.mean(np.power(true-predict, 2))

def mae(true, predict):
    return np.mean(np.abs(true-predict))

def mape(true, predict):
    return np.mean(np.abs(true-predict)/np.abs(true))

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

def StabilityTest(data):
    print('平稳性ADF单位根检验结果如下:')
    print(60*'*')
    print(tsa.adfuller(data))
                   
if __name__ == "__main__":
                   
    Data = pd.read_csv("../2015-2017beijing.csv")
    # 对缺失值进行处理,先对缺失值进行差分处理
    Data = Data.interpolate()
    # 在对缺失值进行向下填充
    Data.fillna(method='bfill', inplace=True)
    weather_data = pd.read_csv("../气候因素/天气数据_小时.csv")
    weather_data.rename(columns={"日期": 'date', '小时': 'hour'}, inplace=True)
    Data = pd.merge(Data, weather_data, how='left', on=['date', 'hour'])
    Data['date'] = pd.to_datetime(Data['date'])
    Data['dayofweek'] = Data['date'].dt.dayofweek
    Data['month'] = Data['date'].dt.month - 1
    Data['wind_x'] = Data['风速']* np.cos(Data['风向角度'] / 180*math.pi)
    Data['wind_y'] = Data['风速'] * np.sin(Data['风向角度'] / 180 * math.pi)
    Data['date'] = pd.date_range(start='2015/01/02 00:00:00', end='2017/12/31 23:00:00', freq='H')
    Data.index = Data['date']
    valid_length = len(Data['2016-11':'2017-05'])
    test_length = len(Data['2017-06':'2017-12'])
    data_train = Data[:'2017-05']
    print("序列的平稳性检验: ")
    print(StabilityTest(Data["pm2.5"]))
    """
    for p in range(0, 25):
        for q in range(0, 1):
            if p==0 and q==0:
                continue
            ParameterSelection(data_train["pm2.5"], p, q)
    """
    # 最优秀的方案选择arima(2, 0, 0)
    print(Data)
    #ets = ETS(data_train["pm2.5"])
    # 开始进行预测
    horizon = 3
    true_matrix = np.zeros((test_length - horizon + 1, horizon))
    predict_matrix = np.zeros((test_length - horizon + 1, horizon))
    """
    for i in tqdm(range(len(Data)-test_length, len(Data)-horizon+1)):
        
        data_new = Data.iloc[:i, 2]
        #print(data_new)
        ets = ETS(data_new)
        true_matrix[i - len(Data) + test_length, :] = np.array(Data.iloc[i:i+horizon, 2])
        predict_matrix[i - len(Data) + test_length, :] = np.array(ets.forecast(horizon))

    np.save("true.npy", true_matrix)
    np.save("pred.npy", predict_matrix)
    for i in [1, 2, 3]:
        print(30 * '#')
        print("%d步预测的mape为:%f" % (i , mape(true_matrix[:, i-1], predict_matrix[:, i-1])))
        print("%d步预测的mae为:%f" % (i , mae(true_matrix[:, i-1], predict_matrix[:, i-1])))
        print("%d步预测的mse为:%f" % (i , mse(true_matrix[:, i-1], predict_matrix[:, i-1])))
        #print("%d步预测da为:%f" % (i, da(true_matrix[:, i - 1], predict_matrix[:, i - 1])))
    """
    result1 = pd.read_excel("ets_horizon_1.xlsx")
    result2 = pd.read_excel("ets_horizon_2.xlsx")
    result3 = pd.read_excel("ets_horizon_3.xlsx")
    print(result1.columns)
    print(result2.columns)
    print(result3.columns)
    print("一步预测的误差")
    print(np.sqrt(test_mse(result1["real value"].values, result1["pred value"].values)))
    print(test_mae(result1["real value"].values, result1["pred value"].values))
    print(test_mape(result1["real value"].values, result1["pred value"].values))
    
    print("二步预测的误差")
    print(np.sqrt(test_mse(result2["real value"].values, result2["pred value"].values)))
    print(test_mae(result2["real value"].values, result2["pred value"].values))
    print(test_mape(result2["real value"].values, result2["pred value"].values))
    
    print("三步预测的误差")
    print(np.sqrt(test_mse(result3["real value"].values, result3["pred value"].values)))
    print(test_mae(result3["real value"].values, result3["pred value"].values))
    print(test_mape(result3["real value"].values, result3["pred value"].values))
        
    
    
        
    
    
    
        
    

        
    
    
    
    
    