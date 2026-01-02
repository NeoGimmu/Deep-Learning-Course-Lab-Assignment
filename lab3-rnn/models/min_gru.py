import numpy as np

class MinGRU:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重 - 使用Xavier初始化
        self.W_z = np.random.randn(hidden_size, hidden_size + input_size) * np.sqrt(2.0 / (hidden_size + input_size))  # 更新门权重
        self.W_r = np.random.randn(hidden_size, hidden_size + input_size) * np.sqrt(2.0 / (hidden_size + input_size))  # 重置门权重
        self.W_h = np.random.randn(hidden_size, hidden_size + input_size) * np.sqrt(2.0 / (hidden_size + input_size))  # 候选隐藏状态权重
        self.W_y = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)  # 输出权重
        
        # 初始化偏置
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # 保存中间变量用于反向传播
        self.reset_grads()
    
    def reset_grads(self):
        # 重置梯度
        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_h = np.zeros_like(self.W_h)
        self.dW_y = np.zeros_like(self.W_y)
        
        self.db_z = np.zeros_like(self.b_z)
        self.db_r = np.zeros_like(self.b_r)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)
        
        # 保存中间变量
        self.combined = None
        self.combined_r = None
        self.z = None
        self.r = None
        self.h_tilde = None
        self.h_prev = None
        self.h = None
        
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev):
        """
        前向传播函数
        x: 当前时间步输入 (input_size, batch_size)
        h_prev: 前一个时间步的隐藏状态 (hidden_size, batch_size)
        返回:
        y: 当前时间步输出 (output_size, batch_size)
        h: 当前时间步的隐藏状态 (hidden_size, batch_size)
        """
        # 拼接隐藏状态和输入
        combined = np.concatenate([h_prev, x], axis=0)  # (hidden_size+input_size, batch_size)
        
        # 更新门：控制是否更新隐藏状态
        z = self.sigmoid(np.dot(self.W_z, combined) + self.b_z)  # (hidden_size, batch_size)
        
        # 重置门：控制是否忽略之前的隐藏状态
        r = self.sigmoid(np.dot(self.W_r, combined) + self.b_r)  # (hidden_size, batch_size)
        
        # 候选隐藏状态
        combined_r = np.concatenate([r * h_prev, x], axis=0)  # (hidden_size+input_size, batch_size)
        h_tilde = self.tanh(np.dot(self.W_h, combined_r) + self.b_h)  # (hidden_size, batch_size)
        
        # 更新隐藏状态：结合前一隐藏状态和候选隐藏状态
        h = (1 - z) * h_prev + z * h_tilde  # (hidden_size, batch_size)
        
        # 输出
        y = np.dot(self.W_y, h) + self.b_y  # (output_size, batch_size)
        
        # 保存中间变量用于反向传播
        self.combined = combined
        self.combined_r = combined_r
        self.z = z
        self.r = r
        self.h_tilde = h_tilde
        self.h_prev = h_prev
        self.h = h
        
        return y, h
    
    def backward(self, dy, dh_next):
        """
        反向传播
        dy: 输出梯度 (output_size, batch_size)
        dh_next: 下一个时间步的隐藏状态梯度 (hidden_size, batch_size)
        返回:
        dx: 输入梯度 (input_size, batch_size)
        dh_prev: 前一个时间步的隐藏状态梯度 (hidden_size, batch_size)
        """
        # 输出梯度
        self.dW_y += np.dot(dy, self.h.T)
        self.db_y += np.sum(dy, axis=1, keepdims=True)
        dh = np.dot(self.W_y.T, dy) + dh_next  # 隐藏状态梯度
        
        # 分解隐藏状态梯度
        dh_tilde = dh * self.z  # 候选隐藏状态梯度
        dz = dh * (self.h_prev - self.h_tilde)  # 更新门梯度
        
        # 更新门梯度
        dz_sigmoid = self.z * (1 - self.z) * dz  # sigmoid导数
        self.dW_z += np.dot(dz_sigmoid, self.combined.T)
        self.db_z += np.sum(dz_sigmoid, axis=1, keepdims=True)
        
        # 候选隐藏状态梯度
        dh_tilde_tanh = (1 - self.h_tilde**2) * dh_tilde  # tanh导数
        self.dW_h += np.dot(dh_tilde_tanh, self.combined_r.T)
        self.db_h += np.sum(dh_tilde_tanh, axis=1, keepdims=True)
        
        # 重置门梯度
        dr_combined = np.dot(self.W_h.T, dh_tilde_tanh)
        dr = dr_combined[:self.hidden_size] * self.h_prev
        dr_sigmoid = self.r * (1 - self.r) * dr  # sigmoid导数
        self.dW_r += np.dot(dr_sigmoid, self.combined.T)
        self.db_r += np.sum(dr_sigmoid, axis=1, keepdims=True)
        
        # 输入梯度和前一个隐藏状态梯度
        dx_combined = np.dot(self.W_z.T, dz_sigmoid) + np.dot(self.W_r.T, dr_sigmoid)
        dx = dx_combined[self.hidden_size:]  # 输入梯度
        dh_prev = dx_combined[:self.hidden_size] + dh * (1 - self.z)  # 前一个隐藏状态梯度
        
        return dx, dh_prev
    
    def update(self, lr):
        # 更新权重和偏置
        self.W_z -= lr * self.dW_z
        self.W_r -= lr * self.dW_r
        self.W_h -= lr * self.dW_h
        self.W_y -= lr * self.dW_y
        
        self.b_z -= lr * self.db_z
        self.b_r -= lr * self.db_r
        self.b_h -= lr * self.db_h
        self.b_y -= lr * self.db_y
        
        # 重置梯度
        self.reset_grads()
