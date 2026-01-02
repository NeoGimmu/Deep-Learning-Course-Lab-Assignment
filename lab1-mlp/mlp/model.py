import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import os

class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None
        # 训练历史（用于可视化）
        self.history = {'train_loss': [],'val_loss': []}
    
    def add_layer(self, layer):
        """添加层（按顺序添加，输入层→隐藏层→输出层）"""
        self.layers.append(layer)
    
    def set_loss(self, loss):
        """设置损失函数"""
        self.loss = loss
    
    def set_optimizer(self, optimizer):
        """设置优化器"""
        self.optimizer = optimizer
    
    def forward(self, x):
        """前向传播：计算模型输出"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_pred, y_true):
        """反向传播：计算梯度"""
        if self.loss is None:
            raise ValueError("请先调用set_loss 设置损失函数")
        grad = self.loss.backward(y_pred, y_true)
        # 反向遍历层（从输出层到输入层）
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self):
        """更新模型参数"""
        if self.optimizer is None:
            raise ValueError("请先调用set_optimizer 设置优化器")
        for layer in self.layers:
            self.optimizer.update(layer)
    
    def train(self, X_train, y_train, epochs=100, batch_size=32,validation_data=None):
        """
        训练模型
        :param X_train: 训练特征(n_samples, n_features)
        :param y_train: 训练标签(n_samples,) 或(n_samples, 1)
        :param epochs: 迭代次数
        :param batch_size: 批次大小
        :param validation_data: 验证集(X_val, y_val)
        :return: 训练历史
        """
        n_samples = X_train.shape[0]
        y_train = y_train.reshape(-1, 1) # 统一形状为(n_samples,1)
        for epoch in range(1, epochs + 1):
            # 打乱训练数据（避免顺序依赖）
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            train_loss = 0.0
            n_batches = n_samples // batch_size
            # 批次训练
            for i in range(n_batches):
                # 取批次数据
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                # 前向传播
                y_pred = self.forward(X_batch)
                # 计算损失
                batch_loss = self.loss.forward(y_pred, y_batch)
                train_loss += batch_loss
                # 反向传播
                self.backward(y_pred, y_batch)
                # 更新参数
                self.update()
            # 计算平均训练损失
            avg_train_loss = train_loss / n_batches
            self.history['train_loss'].append(avg_train_loss)
            # 计算验证损失（如有验证集）
            val_loss = None
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val = y_val.reshape(-1, 1)
                y_val_pred = self.forward(X_val)
                val_loss = self.loss.forward(y_val_pred, y_val)
                self.history['val_loss'].append(val_loss)
            # 每10 个epoch 打印一次日志
            if epoch % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {avg_train_loss:.4f}")
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        :return: (test_loss, metrics) → metrics 包含MAE、R²
        """
        X_test = X_test.reshape(-1, X_test.shape[-1])
        y_test = y_test.reshape(-1, 1)
        y_pred = self.forward(X_test).reshape(-1, 1)
        # 计算损失和指标
        test_loss = self.loss.forward(y_pred, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = {'mae': mae,'r2': r2}
        return test_loss, metrics
    
    def predict(self, X):
        """预测新数据"""
        return self.forward(X).reshape(-1) # 输出形状为(n_samples,)
    
    def plot_history(self,
        save_path='results/training_curve.png'):
        """绘制训练曲线（训练损失vs 验证损失）"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(self.history['train_loss'], label='Train Loss',
        linewidth=2)
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss',linewidth=2, linestyle='--')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training & Validation Loss Curve', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    def plot_predictions(self, X_test, y_test,
        save_path='results/predictions.png'):
        """绘制真实值vs 预测值散点图"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        y_pred = self.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, s=50)
        # 绘制理想预测线（y=x）
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--',
        linewidth=2, label='Ideal Prediction (y=x)')
        plt.xlabel('True Housing Price', fontsize=12)
        plt.ylabel('Predicted Housing Price', fontsize=12)
        plt.title('True vs Predicted Housing Prices', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()