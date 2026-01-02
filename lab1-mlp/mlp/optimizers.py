import numpy as np
class Optimizer:
    """优化器基类：更新层参数"""
    def update(self, layer):
        raise NotImplementedError("子类必须实现update 方法")

class SGD(Optimizer):
    """随机梯度下降：W = W - lr * grad + momentum"""
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        # 保存动量（避免每次创建新变量）
        self.v_w = {} # 权重动量：key=layer_id, value=动量值
        self.v_b = {} # 偏置动量
    
    def update(self, layer):
        # 仅更新有权重和偏置的层（如Dense 层）
        if not hasattr(layer, 'weights') or not hasattr(layer,'biases'):
            return
        
        layer_id = id(layer) # 唯一标识层
        # 初始化动量（首次更新时）
        if layer_id not in self.v_w:
            self.v_w[layer_id] = np.zeros_like(layer.grad_weights)
            self.v_b[layer_id] = np.zeros_like(layer.grad_biases)
        # 计算动量：v = momentum * v_prev - lr * grad
        self.v_w[layer_id] = self.momentum * self.v_w[layer_id] - self.learning_rate * layer.grad_weights
        self.v_b[layer_id] = self.momentum * self.v_b[layer_id] - self.learning_rate * layer.grad_biases

        # 更新参数
        layer.weights += self.v_w[layer_id]
        layer.biases += self.v_b[layer_id]

class Adam(Optimizer):
    """Adam 优化器：结合动量和自适应学习率（默认首选）"""
    def __init__(self, learning_rate=0.001, beta1=0.9,beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0 # 迭代次数（用于偏差修正）
        # 保存一阶动量（m）和二阶动量（v）
        self.m_w = {}
        self.m_b = {}
        self.v_w = {}
        self.v_b = {}
    def update(self, layer):
        if not hasattr(layer, 'weights') or not hasattr(layer,'biases'):
            return
        
        layer_id = id(layer)
        self.t += 1 # 每次更新迭代次数+1

        # 初始化动量
        if layer_id not in self.m_w:
            self.m_w[layer_id] = np.zeros_like(layer.grad_weights)
            self.m_b[layer_id] = np.zeros_like(layer.grad_biases)
            self.v_w[layer_id] = np.zeros_like(layer.grad_weights)
            self.v_b[layer_id] = np.zeros_like(layer.grad_biases)
        # 一阶动量更新：m = beta1*m_prev + (1-beta1)*grad
        self.m_w[layer_id] = self.beta1 * self.m_w[layer_id] + (1 - self.beta1) * layer.grad_weights
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * layer.grad_biases
        # 二阶动量更新：v = beta2*v_prev + (1-beta2)*grad²
        self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1 - self.beta2) * (layer.grad_weights**2)
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (layer.grad_biases**2)
        # 偏差修正（初期m 和v 接近0，修正后更稳定）
        m_w_hat = self.m_w[layer_id] / (1 - self.beta1**self.t)
        m_b_hat = self.m_b[layer_id] / (1 - self.beta1**self.t)
        v_w_hat = self.v_w[layer_id] / (1 - self.beta2**self.t)
        v_b_hat = self.v_b[layer_id] / (1 - self.beta2**self.t)
        # 更新参数：W = W - lr * m_hat / (sqrt(v_hat) + eps)
        layer.weights -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
        layer.biases -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)