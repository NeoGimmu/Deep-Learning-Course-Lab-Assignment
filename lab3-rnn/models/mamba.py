import numpy as np

class Mamba:
    def __init__(self, input_size, hidden_size, output_size, state_size=64, kernel_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size
        self.kernel_size = kernel_size
        
        # 初始化权重 - 使用Xavier初始化
        self.W_in = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        
        # 状态空间参数
        self.A = np.ones((state_size, 1)) * -1.0  # 固定为负值实现稳定状态衰减
        self.B = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / hidden_size)
        self.C = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / state_size)
        self.D = np.random.randn(hidden_size, 1) * 0.1
        
        # 输出投影
        self.W_out = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros((output_size, 1))
        
        # 重置梯度
        self.reset_grads()
        
        # 保存历史梯度用于调试
        self.grad_history = []
    
    def reset_grads(self):
        # 重置梯度
        self.dW_in = np.zeros_like(self.W_in)
        self.dB = np.zeros_like(self.B)
        self.dC = np.zeros_like(self.C)
        self.dD = np.zeros_like(self.D)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)
        
        # 保存中间变量用于反向传播
        self.s_history = []
        self.gate_history = []
        self.x_proj = None
        self.y = None
        self.output = None
        self.x = None
    
    def silu(self, x):
        """SiLU激活函数"""
        return x * (1.0 / (1.0 + np.exp(-x)))
    
    def silu_derivative(self, x):
        """SiLU激活函数的导数"""
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        return sigmoid + x * sigmoid * (1 - sigmoid)
    
    def forward(self, x):
        """
        前向传播函数
        x: 输入序列 (seq_len, input_size, batch_size)
        返回:
        output: 输出序列 (seq_len, output_size, batch_size)
        """
        seq_len, input_size, batch_size = x.shape
        
        # 1. 输入投影：将输入映射到隐藏空间
        x_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            x_proj[t] = np.dot(self.W_in, x[t])
        
        # 2. 初始化状态
        s = np.zeros((self.state_size, batch_size))  # 状态空间初始化
        y = np.zeros((seq_len, self.hidden_size, batch_size))  # 隐藏输出
        
        # 3. 重置历史记录
        self.s_history = []
        self.gate_history = []
        self.x_proj = x_proj
        
        # 4. 处理每个时间步的输入
        for t in range(seq_len):
            # 4.1 选通门：使用SiLU激活函数
            gate = self.silu(x_proj[t])
            
            # 4.2 状态更新：使用状态空间模型
            s = s * np.exp(self.A) + np.dot(self.B.T, gate)  # 状态衰减 + 输入注入
            
            # 4.3 隐藏输出：状态投影 + 选通
            y[t] = gate * (np.dot(self.C, s) + self.D * x_proj[t])
            
            # 4.4 保存中间变量
            self.s_history.append(s.copy())
            self.gate_history.append(gate.copy())
        
        # 5. 输出投影
        output = np.zeros((seq_len, self.output_size, batch_size))
        for t in range(seq_len):
            output[t] = np.dot(self.W_out, y[t]) + self.b_out
        
        # 6. 保存输出用于反向传播
        self.y = y
        self.output = output
        self.x = x
        
        return output
    
    def backward(self, dout):
        """
        反向传播函数
        dout: 输出梯度 (seq_len, output_size, batch_size)
        返回:
        dx: 输入梯度 (seq_len, input_size, batch_size)
        """
        seq_len, output_size, batch_size = dout.shape
        
        # 1. 输出投影的梯度
        dy = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            self.dW_out += np.dot(dout[t], self.y[t].T)
            self.db_out += np.sum(dout[t], axis=1, keepdims=True)
            dy[t] = np.dot(self.W_out.T, dout[t])
        
        # 2. 状态空间的梯度
        ds = np.zeros((self.state_size, batch_size))  # 初始化下一个状态的梯度
        dx_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        
        # 从后往前处理每个时间步
        for t in reversed(range(seq_len)):
            # 当前时间步的中间变量
            gate = self.gate_history[t]
            s_prev = self.s_history[t-1] if t > 0 else np.zeros((self.state_size, batch_size))
            x_proj_t = self.x_proj[t]
            
            # 2.1 计算选通门的梯度
            dgate = dy[t] * (np.dot(self.C, self.s_history[t]) + self.D * x_proj_t)
            dgate += gate * self.D * dx_proj[t]
            dgate *= self.silu_derivative(x_proj_t)
            
            # 2.2 计算D的梯度
            self.dD += np.sum(dy[t] * gate * x_proj_t, axis=1, keepdims=True)
            
            # 2.3 计算C的梯度
            self.dC += np.dot(dy[t] * gate, self.s_history[t].T)
            
            # 2.4 计算状态s的梯度
            ds = np.dot(self.C.T, dy[t] * gate) + ds * np.exp(self.A)
            
            # 2.5 计算B的梯度
            self.dB += np.dot(gate, ds.T)
            
            # 2.6 计算x_proj的梯度
            dx_proj[t] += np.dot(self.B, ds)
            dx_proj[t] += dgate
        
        # 3. 输入投影的梯度
        dx = np.zeros((seq_len, self.input_size, batch_size))
        for t in range(seq_len):
            self.dW_in += np.dot(dx_proj[t], self.x[t].T)
            dx[t] = np.dot(self.W_in.T, dx_proj[t])
        
        # 保存梯度历史
        self.grad_history.append({
            'dW_in': np.linalg.norm(self.dW_in),
            'dB': np.linalg.norm(self.dB),
            'dC': np.linalg.norm(self.dC),
            'dD': np.linalg.norm(self.dD),
            'dW_out': np.linalg.norm(self.dW_out)
        })
        
        return dx
    
    def update(self, lr, weight_decay=0.0):
        # 更新权重和偏置，可选L2正则化
        self.W_in -= lr * (self.dW_in + weight_decay * self.W_in)
        self.B -= lr * (self.dB + weight_decay * self.B)
        self.C -= lr * (self.dC + weight_decay * self.C)
        self.D -= lr * (self.dD + weight_decay * self.D)
        self.W_out -= lr * (self.dW_out + weight_decay * self.W_out)
        self.b_out -= lr * self.db_out
        
        # 重置梯度
        self.reset_grads()
