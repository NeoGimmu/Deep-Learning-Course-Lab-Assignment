import numpy as np
class Loss:
    """损失函数基类"""
    def forward(self, y_pred, y_true):
        raise NotImplementedError("子类必须实现forward 方法")
    def backward(self, y_pred, y_true):
        raise NotImplementedError("子类必须实现backward 方法")

class MSE(Loss):
    """均方误差损失：L = (1/N) * Σ(y_pred - y_true)²（回归任务）"""
    def forward(self, y_pred, y_true):
    # 确保y_pred 和y_true 形状一致（(batch_size, 1)）
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        return np.mean((y_pred - y_true)**2)
    
    def backward(self, y_pred, y_true):
    # 梯度：2*(y_pred - y_true)/N
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        return 2 * (y_pred - y_true) / y_pred.shape[0]