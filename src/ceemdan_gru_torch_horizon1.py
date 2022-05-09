# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
import time
from torch.utils import data
import random
import matplotlib.pyplot as pyplot
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def data_main(df, chosen_scale, lag, horizon, test_length):
    data = df.copy()
    #首先确定测试集的长度
    x_con = np.array(data[['pm2.5', 'pm10', 'so2', 'no2', 'o3', 'co', '温度', '湿度', '降雨量', 'wind_x', 'wind_y', '大气压', 'Imf1','Imf2','Imf3','Imf4','Imf5','Imf6','Imf7','Imf8','Imf9','Imf10','Imf11', 'Imf12', 'Imf13', 'Imf14','Imf15', 'Residual']])
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
    weather_mapping = {'晴': 0, '晴间多云': 1, '多云': 2, '阴': 3, '小雪': 4, '雨夹雪': 5, '小雨': 6, '毛毛雨/细雨': 7, '阵雨': 8, '强阵雨': 9, '雷阵雨': 10, '中雨': 11, '大雨': 12, '薄雾': 13, '中雪': 14, '霾': 15, '雾': 16, '阵雪': 17, '浮尘': 18, '扬沙': 19}
    data['天气状况名称'] = data['天气状况名称'].map(weather_mapping)
    wind_mapping = {'西北风': 0, '西风': 1, '西南风': 2, '东南风': 3, '南风': 4, '东风': 5, '东北风': 6, '北风': 7, '无持续风向': 8}
    data['风向'] = data['风向'].map(wind_mapping)
    length = len(data) - lag - horizon + 1
    x_matrix = np.zeros((length, lag, 28))
    y_matrix = np.zeros((length, 1))
    time_embedding = np.zeros((length, lag+1, 3))
    weather_embedding = np.zeros((length, lag))
    for i in range(length):
        x_matrix[i, :, :] = x_con_nor[i:i + lag, :]
        # 这个地方要注意周期的问题
        y_matrix[i, :] = x_con_nor[i + lag + horizon - 1, 0]
        time_embedding[i, :, :] = np.vstack((np.array(data[['hour', 'dayofweek', 'month']])[i:i + lag, :], np.array(data[['hour', 'dayofweek', 'month']])[i + lag + horizon - 1, :]))
        weather_embedding[i, :] = np.array(data[['天气状况名称']])[i:i + lag].reshape(1,-1)
    return scale, x_matrix, y_matrix, time_embedding, weather_embedding

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.mean_device = torch.tensor(mean[0], dtype=torch.float32).to(device)
        self.std_device = torch.tensor(std[0], dtype=torch.float32).to(device)
    def transform(self, data):
        return (data - self.mean) / self.std
    #主要用于后面进行单变量的损失函数操作
    def inverse_transform(self, data):
        return data * self.std[0] + self.mean[0]
    def inverse_trainform_torch(self, data):
        return data * self.std_device + self.mean_device
#进行归一化
class NormalizedScaler:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.min_device = torch.tensor(min[0], dtype=torch.float32).to(device)
        self.max_device = torch.tensor(max[0], dtype=torch.float32).to(device)
        print(min)
        print(self.min_device)
    def transform(self, data):
        return (data - self.min) / (self.max - self.min)
    def inverse_transform(self, data):
        return data * (self.max[0] - self.min[0]) + self.min[0]
    def inverse_trainform_torch(self, data):
        return data * (self.max_device - self.min_device) + self.min_device

class GruModel(nn.Module):
    def __init__(self, input_size, nhour_embedding_size, wday_embedding_size, season_embedding_size, weather_embedding_size, output_size, hidden_size, num_layer, dropout=0.0):
        super(GruModel, self).__init__()

        self.gru = nn.GRU(input_size+nhour_embedding_size+wday_embedding_size+season_embedding_size+weather_embedding_size, hidden_size, num_layers=num_layer)
        self.output_dense = nn.Linear(hidden_size+nhour_embedding_size+wday_embedding_size+season_embedding_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.nHour_embedding = nn.Embedding(24, nhour_embedding_size)
        self.wday_embedding = nn.Embedding(7, wday_embedding_size)
        self.season_embedding = nn.Embedding(12, season_embedding_size)
        self.weather_embedding = nn.Embedding(20, weather_embedding_size)
        #self.init_weights()

    def init_weights(self):
        #self.output_dense.bias.data.fill_(0)
        nn.init.xavier_uniform(self.output_dense.weight, gain=np.sqrt(2))
        nn.init.constant(self.output_dense.bias, 0)

    def forward(self, x, time_embedding, weather_embedding):
        hour_embedding = time_embedding[:,:,0]
        day_embedding = time_embedding[:,:,1]
        season_embedding = time_embedding[:,:,2]
        hour_embedding = self.nHour_embedding(hour_embedding)
        day_embedding = self.wday_embedding(day_embedding)
        season_embedding = self.season_embedding(season_embedding)
        weather_embedding_encoder = self.weather_embedding(weather_embedding)
        #定义encoder部分的外生变量
        hour_embedding_encoder = hour_embedding[:,:-1,:]
        day_embedding_encoder = day_embedding[:,:-1,:]
        season_embedding_encoder = season_embedding[:,:-1,:]

        hour_embedding_decoder = hour_embedding[:,-1,:]
        day_embedding_decoder = day_embedding[:, -1, :]
        season_embedding_decoder = season_embedding[:, -1, :]

        x_cat = torch.cat([x, hour_embedding_encoder, day_embedding_encoder, season_embedding_encoder, weather_embedding_encoder], dim=-1)
        x_input = x_cat.permute(1, 0, 2)
        y,_ = self.gru(x_input)
        y = torch.cat([y[-1, :, :], hour_embedding_decoder, day_embedding_decoder, season_embedding_decoder], dim=-1)
        y = self.dropout(y)
        y = self.output_dense(y)
        return y


class Mask_rmse_loss(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, y_true, y_pred):
        y_true_inv = self.scaler.inverse_trainform_torch(y_true)
        y_pred_inv = self.scaler.inverse_trainform_torch(y_pred)
        mask = (y_true_inv != 0).float()
        mask /= mask.mean()
        loss = torch.pow((y_true_inv - y_pred_inv), 2)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return torch.sqrt(loss.mean())


class Mse_loss(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, y_true, y_pred):
        y_true_inv = self.scaler.inverse_trainform_torch(y_true)
        y_pred_inv = self.scaler.inverse_trainform_torch(y_pred)
        mask = (y_true_inv != 0).float()
        mask /= mask.mean()
        loss = torch.pow((y_true_inv - y_pred_inv), 2)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()


class Mae_loss(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, y_true, y_pred):
        y_true_inv = self.scaler.inverse_trainform_torch(y_true)
        y_pred_inv = self.scaler.inverse_trainform_torch(y_pred)
        mask = (y_true_inv != 0).float()
        mask /= mask.mean()
        loss = torch.abs(y_true_inv - y_pred_inv)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()


class Mape_loss(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, y_true, y_pred):
        y_true_inv = self.scaler.inverse_trainform_torch(y_true)
        y_pred_inv = self.scaler.inverse_trainform_torch(y_pred)
        mask = (y_true_inv != 0).float()
        mask /= mask.mean()
        loss = torch.abs(y_true_inv - y_pred_inv) / torch.abs(y_true_inv)
        loss = loss * mask
        # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
        loss[loss != loss] = 0
        return loss.mean()


# 输出参数的个数
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def train(model, data_iter, val_x, val_time_embedding, val_weather_embedding, val_y, optimizer, num_epochs, device,
          scaler, horizon):
    model.to(device)
    loss = Mape_loss(scaler)
    mae_loss = Mae_loss(scaler)
    mse_loss = Mse_loss(scaler)
    mape_loss = Mape_loss(scaler)
    train_loss = []
    val_loss = []
    val_l = 1e8
    for epoch in range(1, num_epochs + 1):
        print("Epoch %d/%d" % (epoch, num_epochs))
        model.train()
        time_start = time.time()
        l_sum = 0.0
        batch_num = 0
        for batch in data_iter:
            batch_num += 1
            optimizer.zero_grad()
            x, time_embedding, weather_embedding, y = [xx.to(device) for xx in batch]
            y_pred = model(x.to(device), time_embedding.to(device), weather_embedding.to(device))
            l = loss(y.to(device), y_pred)
            l.backward()
            optimizer.step()
            l_sum += float(l.sum().cpu().item())
        with torch.no_grad():
            model.eval()
            y_val_pred = model(val_x.to(device), val_time_embedding.to(device), val_weather_embedding.to(device))
            l_val = loss(val_y.to(device), y_val_pred)
            l_val_sum = float(l_val.sum().cpu().item())
        train_loss.append(l_sum / batch_num)
        val_loss.append(l_val_sum)
        val_mse = mse_loss(val_y.to(device), y_val_pred)
        val_mae = mae_loss(val_y.to(device), y_val_pred)
        val_mape = mape_loss(val_y.to(device), y_val_pred)
        val_mape_cpu = float(val_mape.sum().cpu().item())
        if val_mape_cpu < val_l:
            torch.save(model, 'model/ceemdan_gru_model_%s.pkl' % (str(horizon)))
            print("val_mape is decrease from %f to %f" % (val_l, val_mape_cpu))
            val_l = val_mape_cpu
        # 对mape比较低的模型进行保存
        '''
        if l_val_sum<val_l:
            torch.save(model, 'model/model_%s.pkl' %(str(horizon)))
            print("val_loss is decrease from %f to %f" %(val_l, l_val_sum))
            val_l = l_val_sum
        '''
        time_final = time.time()
        # 对模型进行保存
        print("train_loss:%f | val_loss:%f| val_mse:%f| val_mae:%f| val_mape:%f| time: %f s" % (
        l_sum / batch_num, l_val_sum, val_mse, val_mae, val_mape, time_final - time_start))
    return train_loss, val_loss, model


if __name__ == "__main__":
    # 先读取数据
    # setup_seed(420)
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
    Data['wind_x'] = Data['风速'] * np.cos(Data['风向角度'] / 180 * math.pi)
    Data['wind_y'] = Data['风速'] * np.sin(Data['风向角度'] / 180 * math.pi)
    # data.to_csv("../data.csv", encoding='GBK')
    # 对数据进行训练集、验证集、测试集划分、按照比例为 6:2:2的比例进行划分
    # 训练集 '2015-01-02: 2016-10-31'
    # 验证集 '2016-11-01: 2017-05-31'
    # 验证集 '2017-06-01: 2017-12-31'
    ceemdan_data = pd.read_excel("分解数据.xlsx")
    Data = pd.merge(Data, ceemdan_data, how='left', on=['date', 'hour'])
    Data.index = Data['date']
    valid_lenth = len(Data['2016-11':'2017-05'])
    test_lenth = len(Data['2017-06':'2017-12'])
    lag = 24
    horizon = 1
    scale, x_matrix, y_matrix, time_embedding, weather_embedding = data_main(Data, 'StandardScaler', lag, horizon,
                                                                             test_lenth)

    x_matrix_train = x_matrix[:-test_lenth, :, :]
    # x_train = x_matrix_train[:-valid_lenth, :, :]
    # x_valid = x_matrix_train[-valid_lenth:, :, :]
    x_train = x_matrix_train
    x_test = x_matrix[-test_lenth:, :, :]

    y_matrix_train = y_matrix[:-test_lenth, :]
    # y_train = y_matrix_train[:-valid_lenth, :]
    # y_valid = y_matrix_train[-valid_lenth:, :]
    y_train = y_matrix_train
    y_test = y_matrix[-test_lenth:, :]

    time_embedding_train_ = time_embedding[:-test_lenth, :, :]
    # time_embedding_train = time_embedding_train_[:-valid_lenth, :, :]
    # time_embedding_valid = time_embedding_train_[-valid_lenth:, :, :]
    time_embedding_train = time_embedding_train_
    time_embedding_test = time_embedding[-test_lenth:, :, :]

    weather_embedding_train_ = weather_embedding[:-test_lenth, :]
    # weather_embedding_train = weather_embedding_train_[:-valid_lenth, :]
    # weather_embedding_valid = weather_embedding_train_[-valid_lenth:, :]
    weather_embedding_train = weather_embedding_train_
    weather_embedding_test = weather_embedding[-test_lenth:, :]

    x_train = torch.tensor(x_train, dtype=torch.float32)
    # x_valid = torch.tensor(x_valid, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.float32)
    # y_valid = torch.tensor(y_valid, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    # 注意 用embedding需要torch.long的类型
    time_embedding_train = torch.tensor(time_embedding_train, dtype=torch.long)
    # time_embedding_valid = torch.tensor(time_embedding_valid, dtype=torch.long)
    time_embedding_test = torch.tensor(time_embedding_test, dtype=torch.long)

    weather_embedding_train = torch.tensor(weather_embedding_train, dtype=torch.long)
    # weather_embedding_valid = torch.tensor(weather_embedding_valid, dtype=torch.long)
    weather_embedding_test = torch.tensor(weather_embedding_test, dtype=torch.long)

    input_size = x_matrix_train.shape[2]
    nhour_embedding_size = 4
    wday_embedding_size = 2
    season_embedding_size = 2
    weather_embedding_size = 2
    output_size = 1

    hidden_size = 64
    dropout = 0
    batch_size = 128
    num_epochs = 200
    num_layer = 1
    model = GruModel(input_size, nhour_embedding_size, wday_embedding_size, season_embedding_size,
                     weather_embedding_size, output_size, hidden_size, num_layer, dropout=dropout)
    print(get_parameter_number(model))
    lr = 0.01
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_data = data.TensorDataset(x_train, time_embedding_train, weather_embedding_train, y_train)
    train_iter = data.DataLoader(train_data, batch_size, shuffle=True)
    '''
    # train_loss, val_loss, model = train(model, train_iter, x_valid, time_embedding_valid, weather_embedding_valid, y_valid, optimizer, num_epochs, device, scale, horizon)
    train_loss, val_loss, model = train(model, train_iter, x_test, time_embedding_test, weather_embedding_test, y_test,
                                        optimizer, num_epochs, device, scale, horizon)
    '''
    mae_loss = Mae_loss(scale)
    mse_loss = Mse_loss(scale)
    mape_loss = Mape_loss(scale)

    model = torch.load('model/ceemdan_gru_model_%s.pkl' % (str(horizon)))
    with torch.no_grad():
        model.to(device)
        model.eval()
        y_test_pred = model(x_test.to(device), time_embedding_test.to(device), weather_embedding_test.to(device))
        test_mse = mse_loss(y_test.to(device), y_test_pred)
        test_mae = mae_loss(y_test.to(device), y_test_pred)
        test_mape = mape_loss(y_test.to(device), y_test_pred)
        print("test_mse:%f| test_mae:%f| test_mape:%f" %(test_mse, test_mae, test_mape))
    y_pre_test_inv = scale.inverse_transform(y_test_pred.cpu().numpy())
    y_true_test_inv = scale.inverse_transform(y_test.cpu().numpy())
    print(np.mean(np.power((y_pre_test_inv - y_true_test_inv), 2)))
    print(np.mean(np.abs(y_pre_test_inv - y_true_test_inv)))
    print(np.mean(np.abs(y_pre_test_inv - y_true_test_inv)/np.abs(y_true_test_inv)))
    data_frame = pd.DataFrame({})
    data_frame['real value'] = y_true_test_inv.reshape(1, -1)[0]
    data_frame['pred value'] = y_pre_test_inv.reshape(1, -1)[0]
    data_frame = pd.DataFrame(data_frame)
    data_frame.to_excel("ceemdan_gru_horizon_%d.xlsx" %horizon, index=False)
    





