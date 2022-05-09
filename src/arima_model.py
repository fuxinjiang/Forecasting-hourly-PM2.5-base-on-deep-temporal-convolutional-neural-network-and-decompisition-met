# %%arima预测
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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

def mse(true, predict):
    return np.mean(np.power(true-predict, 2))

def mae(true, predict):
    return np.mean(np.abs(true-predict))

def mape(true, predict):
    return np.mean(np.abs(true-predict)/np.abs(true))

def StabilityTest(data):
    print('平稳性ADF单位根检验结果如下:')
    print(60*'*')
    print(tsa.adfuller(data))
    
# %%由于数据较多，我们采用多线程的角度进行预测 分析
def main1(model, data):
    
    user_app_class_index={}
    
    def subpro(uni_class):
        app_class_list = app_class_dict[uni_class]
        duration_list = []
        for i in app_class_list:
            if i in effective_app_id:
                index = effective_app_id.index(i)
                duration_list.append(int(index))
        app_class_index[uni_class] = duration_list

    with multiprocessing.Manager() as manager:
        app_class_index = manager.dict()
    #进行多进程对上面的进行计算解决
        multipro = []
        #选取有用的APP类别，去除没有用的APP类别，加快速度
        #useful_attri = list(set(app_info['app_class'])-set(['合作壁纸*', '休闲娱乐', '模拟游戏', '角色游戏', '主题铃声', '策略游戏', '医疗健康', '体育射击', '电子书籍', '动作冒险']))
        for i, class_name in enumerate(useful_attri):
            #定义进程的名字
            thread_name = "thead_%d" % i
            multipro.append(multiprocessing.Process(target=subpro, name=thread_name, args=(class_name, )))
        for process in multipro:
            process.start()
        for process in multipro:
            process.join()
        print("多进程计算完毕!")

        for class_name in useful_attri:
            user_app_class_index[class_name] = app_class_index[class_name]
    sparse_matrix_train_class = np.zeros((sparse_matrix_train.shape[0], len(useful_attri)))
    sparse_matrix_test_class = np.zeros((sparse_matrix_test.shape[0], len(useful_attri)))
    for i, class_name in enumerate(useful_attri):
        sparse_matrix_train_class[:, i] = get_sparse_matrix_sum(user_app_class_index[class_name], sparse_matrix_train)
        sparse_matrix_test_class[:, i] = get_sparse_matrix_sum(user_app_class_index[class_name], sparse_matrix_test)

    np.save("train_sparse_matrix_%s_sum_app_class.npy" % optation, sparse_matrix_train_class)
    np.save("test_sparse_matrix_%s_sum_app_class.npy" % optation, sparse_matrix_test_class)
    print("结束！")

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
    arima_model, _, _ = ParameterSelection(data_train["pm2.5"], 1, 1)
    # 开始进行预测
    horizon = 3
    true_matrix = np.zeros((test_length - horizon + 1, horizon))
    predict_matrix = np.zeros((test_length - horizon + 1, horizon))
    """
    for i in tqdm(range(len(Data)-test_length, len(Data)-horizon+1)):
        
        data_new = Data.iloc[:i, 2]
        model_new = arima_model.apply(data_new)
        true_matrix[i - len(Data) + test_length, :] = np.array(Data.iloc[i:i+horizon, 2])
        predict_matrix[i - len(Data) + test_length, :] = np.array(model_new.forecast(horizon))
        
    for i in [1, 2, 3]:
        print(30 * '#')
        print("%d步预测的mape为:%f" % (i , mape(true_matrix[:, i-1], predict_matrix[:, i-1])))
        print("%d步预测的mae为:%f" % (i , mae(true_matrix[:, i-1], predict_matrix[:, i-1])))
        print("%d步预测的mse为:%f" % (i , mse(true_matrix[:, i-1], predict_matrix[:, i-1])))
        #print("%d步预测da为:%f" % (i, da(true_matrix[:, i - 1], predict_matrix[:, i - 1])))

    for i in [1, 2, 3]:
        arima_df = pd.DataFrame({})
        arima_df["real value"] = true_matrix[:, i-1]
        arima_df["pred value"] = predict_matrix[:, i-1]
        arima_df.to_excel("arima_horizon_%d.xlsx" %i, index=False)
    """
    result1 = pd.read_excel("arima_horizon_1.xlsx")
    result2 = pd.read_excel("arima_horizon_2.xlsx")
    result3 = pd.read_excel("arima_horizon_3.xlsx")
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
    
    
        
    
    
    
        
    

        
    
    
    
    
    