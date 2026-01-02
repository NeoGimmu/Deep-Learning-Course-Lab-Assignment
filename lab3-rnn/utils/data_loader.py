import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

class DataLoader:
    def __init__(self, data_dir='labs-rnn/data'):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_yahoo_stock(self, ticker='AAPL', start_date='2010-01-01', end_date='2023-12-31'):
        """
        加载雅虎股票数据
        """
        try:
            # 先尝试使用本地文件
            local_file = '../AAPL_stock_data.csv'
            if os.path.exists(local_file):
                print(f"Using local data file: {local_file}")
                df = pd.read_csv(local_file, index_col=0, parse_dates=True)
            else:
                # 下载数据
                df = yf.download(ticker, start=start_date, end=end_date)
                
                # 保存数据到本地
                df.to_csv(local_file)
            
            # 只使用收盘价
            data = df['Close'].values.reshape(-1, 1)
            
            # 数据归一化
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            return {
                'data': scaled_data,
                'scaler': scaler,
                'original_data': data
            }
        except Exception as e:
            print(f"Error loading stock data: {e}")
            raise
    
    def create_stock_batches(self, data, seq_len, batch_size):
        """
        创建股票数据批次
        
        参数:
        data: 归一化后的股票数据 (n_samples, 1)
        seq_len: 序列长度，使用前seq_len个值预测下一个值
        batch_size: 批次大小
        
        返回:
        batches: 批次列表，每个批次是(x, y)元组
        x: (seq_len, input_size, batch_size)
        y: (1, batch_size)
        """
        total_len = len(data)
        x = []
        y = []
        
        # 创建序列
        for i in range(total_len - seq_len):
            x.append(data[i:i+seq_len])  # 前seq_len个值
            y.append(data[i+seq_len])  # 下一个值
        
        x = np.array(x)
        y = np.array(y)
        
        # 确保批次完整
        n_batches = len(x) // batch_size
        x = x[:n_batches * batch_size]
        y = y[:n_batches * batch_size]
        
        # 转换为(batch_size, n_batches, seq_len, 1)然后转置为(seq_len, 1, batch_size)
        x = x.reshape(batch_size, n_batches, seq_len, 1).transpose(2, 3, 0, 1)
        y = y.reshape(batch_size, n_batches, 1).transpose(2, 0, 1)
        
        batches = []
        for i in range(n_batches):
            batches.append((x[:, :, :, i], y[:, :, i]))
        
        return batches
    
    def split_data(self, data, train_ratio=0.8):
        """
        将数据分为训练集和验证集
        """
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        valid_data = data[train_size:]
        return train_data, valid_data
