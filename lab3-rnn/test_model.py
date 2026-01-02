import numpy as np
import time
from utils.data_loader import DataLoader
from models.min_gru import MinGRU
from models.mamba import Mamba

# 定义损失函数
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.shape[1]  # 除以批次大小

def train_min_gru():
    """训练MinGRU模型"""
    # 加载数据
    loader = DataLoader()
    stock_data = loader.load_yahoo_stock(ticker='AAPL')
    data = stock_data['data']
    scaler = stock_data['scaler']
    
    # 划分训练集和验证集
    train_data, valid_data = loader.split_data(data, train_ratio=0.8)
    
    # 超参数
    input_size = 1
    hidden_size = 128
    output_size = 1
    seq_len = 20
    batch_size = 32
    learning_rate = 0.01
    epochs = 5
    
    # 创建批次
    train_batches = loader.create_stock_batches(train_data, seq_len, batch_size)
    valid_batches = loader.create_stock_batches(valid_data, seq_len, batch_size)
    
    # 创建模型
    model = MinGRU(input_size, hidden_size, output_size)
    
    train_losses = []
    valid_losses = []
    
    print("Training MinGRU...")
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_train_loss = 0.0
        
        # 训练
        for x_batch, y_batch in train_batches:
            h_prev = np.zeros((hidden_size, batch_size))
            batch_loss = 0.0
            
            # 前向传播
            outputs = []
            for t in range(seq_len):
                y_pred, h_prev = model.forward(x_batch[t], h_prev)
                outputs.append(y_pred)
            
            # 计算损失
            y_pred = outputs[-1]  # 最后一个时间步的输出
            loss = mse_loss(y_pred, y_batch)
            epoch_train_loss += loss
            
            # 反向传播
            dy = mse_loss_derivative(y_pred, y_batch)
            dh_next = np.zeros((hidden_size, batch_size))
            
            # 从后往前反向传播
            for t in reversed(range(seq_len)):
                dy, dh_next = model.backward(dy, dh_next)
            
            # 更新参数
            model.update(learning_rate)
        
        # 验证
        epoch_valid_loss = 0.0
        for x_batch, y_batch in valid_batches:
            h_prev = np.zeros((hidden_size, batch_size))
            
            # 前向传播
            for t in range(seq_len):
                y_pred, h_prev = model.forward(x_batch[t], h_prev)
            
            # 计算损失
            loss = mse_loss(y_pred, y_batch)
            epoch_valid_loss += loss
        
        # 平均损失
        avg_train_loss = epoch_train_loss / len(train_batches)
        avg_valid_loss = epoch_valid_loss / len(valid_batches)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, MinGRU Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}, Time: {epoch_time:.2f}s")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'scaler': scaler
    }

def train_mamba():
    """训练Mamba模型"""
    # 加载数据
    loader = DataLoader()
    stock_data = loader.load_yahoo_stock(ticker='AAPL')
    data = stock_data['data']
    scaler = stock_data['scaler']
    
    # 划分训练集和验证集
    train_data, valid_data = loader.split_data(data, train_ratio=0.8)
    
    # 超参数
    input_size = 1
    hidden_size = 128
    output_size = 1
    state_size = 64
    seq_len = 20
    batch_size = 32
    learning_rate = 0.01
    epochs = 5
    
    # 创建批次
    train_batches = loader.create_stock_batches(train_data, seq_len, batch_size)
    valid_batches = loader.create_stock_batches(valid_data, seq_len, batch_size)
    
    # 创建模型
    model = Mamba(input_size, hidden_size, output_size, state_size=state_size)
    
    train_losses = []
    valid_losses = []
    
    print("\nTraining Mamba...")
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_train_loss = 0.0
        
        # 训练
        for x_batch, y_batch in train_batches:
            # 前向传播
            outputs = model.forward(x_batch)
            
            # 计算损失 (只使用最后一个时间步的输出)
            y_pred = outputs[-1]
            loss = mse_loss(y_pred, y_batch)
            epoch_train_loss += loss
            
            # 反向传播
            dy = mse_loss_derivative(y_pred, y_batch)
            dout = np.zeros_like(outputs)
            dout[-1] = dy  # 只有最后一个时间步有损失
            
            # 反向传播
            model.backward(dout)
            
            # 更新参数
            model.update(learning_rate)
        
        # 验证
        epoch_valid_loss = 0.0
        for x_batch, y_batch in valid_batches:
            # 前向传播
            outputs = model.forward(x_batch)
            y_pred = outputs[-1]
            
            # 计算损失
            loss = mse_loss(y_pred, y_batch)
            epoch_valid_loss += loss
        
        # 平均损失
        avg_train_loss = epoch_train_loss / len(train_batches)
        avg_valid_loss = epoch_valid_loss / len(valid_batches)
        
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Mamba Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}, Time: {epoch_time:.2f}s")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'scaler': scaler
    }

if __name__ == "__main__":
    # 选择要训练的模型类型
    model_type = 'mamba'  # 'min_gru' or 'mamba'
    
    if model_type == 'min_gru':
        train_min_gru()
    elif model_type == 'mamba':
        train_mamba()
    else:
        print("Invalid model type. Choose 'min_gru' or 'mamba'.")
