import numpy as np
from .layers import Layer
class Activation(Layer):
    """激活函数基类：保存输入用于反向传播"""
    def __init__(self):
        self.input = None

class ReLU(Activation):
    """ReLU 激活函数：f(x) = max(0, x)（隐藏层首选）"""
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    def backward(self, grad_output):
        """梯度：x>0 时为1，否则为0"""
        grad_x = np.where(self.input > 0, 1, 0)
        return grad_x * grad_output
    
class Tanh(Activation):
    """Tanh 激活函数：f(x) = (e^x - e^(-x))/(e^x + e^(-x))"""
    def forward(self, x):
        self.input = x
        return np.tanh(x)
    def backward(self, grad_output):
        """梯度：1 - tanh²(x)"""
        return grad_output * (1 - np.tanh(self.input)**2)

class Sigmoid(Activation):
    """Sigmoid 激活函数：f(x) = 1/(1+e^(-x))（二分类输出层）"""
    def forward(self, x):
        self.input = x
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # 避免指数溢出
    def backward(self, grad_output):
        """梯度：sigmoid(x) * (1 - sigmoid(x))"""
        sig = self.forward(self.input)
        return grad_output * sig * (1 - sig)

class Linear(Activation):
    """线性激活函数：f(x) = x（回归任务输出层）"""
    def forward(self, x):
        self.input = x
        return x
    
    def backward(self, grad_output):
        """梯度：1（直接传递梯度）"""
        return grad_output