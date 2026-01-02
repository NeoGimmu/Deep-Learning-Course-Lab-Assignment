import numpy as np
import time
import matplotlib.pyplot as plt
from utils.data_loader import DataLoader
from models.min_gru import MinGRU
from models.mamba import Mamba

# 定义损失函数
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.shape[1]  # 除以批次大小

def rmse_loss(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def r2_score(y_pred, y_true):
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

def train_model(model_type, train_data, valid_data, seq_len, batch_size, learning_rate, epochs):
    """训练模型"""
    # 创建批次
    loader = DataLoader()
    train_batches = loader.create_stock_batches(train_data, seq_len, batch_size)
    valid_batches = loader.create_stock_batches(valid_data, seq_len, batch_size)
    
    # 创建模型
    input_size = 1
    hidden_size = 128
    output_size = 1
    
    if model_type == 'min_gru':
        model = MinGRU(input_size, hidden_size, output_size)
    elif model_type == 'mamba':
            model = Mamba(input_size, hidden_size, output_size, state_size=128)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        epoch_train_loss = 0.0
        
        for x_batch, y_batch in train_batches:
            if model_type == 'min_gru':
                h_prev = np.zeros((hidden_size, batch_size))
                
                # 前向传播
                outputs = []
                for t in range(seq_len):
                    y_pred, h_prev = model.forward(x_batch[t], h_prev)
                    outputs.append(y_pred)
                
                # 计算损失
                y_pred = outputs[-1]
                loss = mse_loss(y_pred, y_batch)
                epoch_train_loss += loss
                
                # 反向传播
                dy = mse_loss_derivative(y_pred, y_batch)
                dh_next = np.zeros((hidden_size, batch_size))
                
                # 从后往前反向传播
                for t in reversed(range(seq_len)):
                    dy, dh_next = model.backward(dy, dh_next)
            
            elif model_type == 'mamba':
                # 前向传播
                outputs = model.forward(x_batch)
                y_pred = outputs[-1]
                
                # 计算损失
                loss = mse_loss(y_pred, y_batch)
                epoch_train_loss += loss
                
                # 反向传播
                dy = mse_loss_derivative(y_pred, y_batch)
                dout = np.zeros_like(outputs)
                dout[-1] = dy
                model.backward(dout)
            
            # 更新参数
            if model_type == 'min_gru':
                model.update(learning_rate)
            elif model_type == 'mamba':
                model.update(learning_rate, weight_decay=0.01)
        
        avg_train_loss = epoch_train_loss / len(train_batches)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        epoch_valid_loss = 0.0
        
        for x_batch, y_batch in valid_batches:
            if model_type == 'min_gru':
                h_prev = np.zeros((hidden_size, batch_size))
                
                # 前向传播
                for t in range(seq_len):
                    y_pred, h_prev = model.forward(x_batch[t], h_prev)
                
                # 计算损失
                loss = mse_loss(y_pred, y_batch)
                epoch_valid_loss += loss
            
            elif model_type == 'mamba':
                # 前向传播
                outputs = model.forward(x_batch)
                y_pred = outputs[-1]
                
                # 计算损失
                loss = mse_loss(y_pred, y_batch)
                epoch_valid_loss += loss
        
        avg_valid_loss = epoch_valid_loss / len(valid_batches)
        valid_losses.append(avg_valid_loss)
        
        # 打印每个epoch的训练损失和验证损失
        print(f"Epoch {epoch+1}/{epochs}, {model_type} Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}")
    
    return model, train_losses, valid_losses

def validate_model(model_type, model, data, seq_len, batch_size, scaler):
    """验证模型"""
    loader = DataLoader()
    batches = loader.create_stock_batches(data, seq_len, batch_size)
    
    all_predictions = []
    all_actual = []
    
    input_size = 1
    hidden_size = 128
    
    for x_batch, y_batch in batches:
        if model_type == 'min_gru':
            h_prev = np.zeros((hidden_size, batch_size))
            
            for t in range(seq_len):
                y_pred, h_prev = model.forward(x_batch[t], h_prev)
        
        elif model_type == 'mamba':
            outputs = model.forward(x_batch)
            y_pred = outputs[-1]
        
        # 保存预测结果和实际值
        all_predictions.append(y_pred)
        all_actual.append(y_batch)
    
    # 合并所有批次的结果
    predictions = np.concatenate(all_predictions, axis=1)
    actual = np.concatenate(all_actual, axis=1)
    
    # 反归一化
    predictions_unscaled = scaler.inverse_transform(predictions.T).T
    actual_unscaled = scaler.inverse_transform(actual.T).T
    
    # 计算评估指标
    mse = mse_loss(predictions_unscaled, actual_unscaled)
    rmse = rmse_loss(predictions_unscaled, actual_unscaled)
    mae = mae_loss(predictions_unscaled, actual_unscaled)
    r2 = r2_score(predictions_unscaled, actual_unscaled)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions_unscaled,
        'actual': actual_unscaled
    }

def main():
    """主函数，运行模型对比实验"""
    # 加载数据
    loader = DataLoader()
    stock_data = loader.load_yahoo_stock(ticker='AAPL')
    data = stock_data['data']
    scaler = stock_data['scaler']
    
    # 划分训练集和验证集
    train_ratio = 0.8
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    valid_data = data[train_size:]
    
    # 超参数
    seq_len = 20
    batch_size = 32
    min_gru_learning_rate = 0.01
    mamba_learning_rate = 0.01
    epochs = 6
    
    print("Running benchmark experiments...")
    
    # 训练和验证MinGRU
    print("\n=== Training MinGRU ===")
    start_time = time.time()
    min_gru_model, min_gru_train_losses, min_gru_valid_losses = train_model('min_gru', train_data, valid_data, seq_len, batch_size, min_gru_learning_rate, epochs)
    min_gru_time = time.time() - start_time
    min_gru_valid = validate_model('min_gru', min_gru_model, valid_data, seq_len, batch_size, scaler)
    
    # 训练和验证Mamba
    print("\n=== Training Mamba ===")
    start_time = time.time()
    mamba_model, mamba_train_losses, mamba_valid_losses = train_model('mamba', train_data, valid_data, seq_len, batch_size, mamba_learning_rate, epochs)
    mamba_time = time.time() - start_time
    mamba_valid = validate_model('mamba', mamba_model, valid_data, seq_len, batch_size, scaler)
    
    # 打印结果
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    
    print("\nTraining Time:")
    print(f"MinGRU: {min_gru_time:.2f}s")
    print(f"Mamba: {mamba_time:.2f}s")
    
    print("\nValidation Metrics:")
    print(f"{'Metric':<10} {'MinGRU':<15} {'Mamba':<15}")
    print("-"*40)
    print(f"{'MSE':<10} {min_gru_valid['mse']:<15.6f} {mamba_valid['mse']:<15.6f}")
    print(f"{'RMSE':<10} {min_gru_valid['rmse']:<15.6f} {mamba_valid['rmse']:<15.6f}")
    print(f"{'MAE':<10} {min_gru_valid['mae']:<15.6f} {mamba_valid['mae']:<15.6f}")
    print(f"{'R2':<10} {min_gru_valid['r2']:<15.6f} {mamba_valid['r2']:<15.6f}")
    
    print("\nTraining Loss (Final Epoch):")
    print(f"MinGRU: {min_gru_train_losses[-1]:.6f}")
    print(f"Mamba: {mamba_train_losses[-1]:.6f}")
    
    # 保存结果
    results = {
        'min_gru': {
            'train_losses': min_gru_train_losses,
            'valid_losses': min_gru_valid_losses,
            'valid_metrics': min_gru_valid,
            'training_time': min_gru_time
        },
        'mamba': {
            'train_losses': mamba_train_losses,
            'valid_losses': mamba_valid_losses,
            'valid_metrics': mamba_valid,
            'training_time': mamba_time
        },
        'hyperparameters': {
            'seq_len': seq_len,
            'batch_size': batch_size,
            'min_gru_learning_rate': min_gru_learning_rate,
            'mamba_learning_rate': mamba_learning_rate,
            'epochs': epochs
        }
    }
    
    # 绘制损失曲线和验证指标
    plt.figure(figsize=(15, 10))
    
    # MinGRU 训练和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs+1), min_gru_train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epochs+1), min_gru_valid_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title('MinGRU Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # Mamba 训练和验证损失
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs+1), mamba_train_losses, label='Training Loss', color='green')
    plt.plot(range(1, epochs+1), mamba_valid_losses, label='Validation Loss', color='orange', linestyle='--')
    plt.title('Mamba Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # 训练损失对比
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs+1), min_gru_train_losses, label='MinGRU Training', color='blue')
    plt.plot(range(1, epochs+1), mamba_train_losses, label='Mamba Training', color='green')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # 验证指标对比
    plt.subplot(2, 2, 4)
    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    min_gru_values = [min_gru_valid['mse'], min_gru_valid['rmse'], min_gru_valid['mae'], min_gru_valid['r2']]
    mamba_values = [mamba_valid['mse'], mamba_valid['rmse'], mamba_valid['mae'], mamba_valid['r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, min_gru_values, width, label='MinGRU')
    plt.bar(x + width/2, mamba_values, width, label='Mamba')
    plt.title('Validation Metrics Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('mamba_vs_mingru.png')
    print("\nResults saved to mamba_vs_mingru.png")
    
    return results

if __name__ == "__main__":
    main()
