import numpy as np
class Layer:
    """层基类：所有层需实现forward/backward 方法"""
    def forward(self, x):
        raise NotImplementedError("子类必须实现forward 方法")

    def backward(self, grad_output):
        raise NotImplementedError("子类必须实现backward 方法")

    def update_params(self, learning_rate):
        """无参数层（如激活函数）无需实现"""
        pass

class Dense(Layer):
    """全连接层：y = W @ x + b"""
    def __init__(self, input_dim, output_dim,weight_initializer='he'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_initializer = weight_initializer
        
        # 权重初始化（关键：影响模型收敛速度）
        if weight_initializer == 'he':
            # He 初始化（适合ReLU 激活）：W ~ N(0, 2/input_dim)
            self.weights = np.random.randn(input_dim, output_dim)* np.sqrt(2 / input_dim)
        elif weight_initializer == 'xavier':
        # Xavier 初始化（适合Tanh/Sigmoid）：W ~ N(0,1/(input_dim+output_dim))
            self.weights = np.random.randn(input_dim, output_dim)* np.sqrt(1 / (input_dim + output_dim))
        else:
            raise ValueError("仅支持he/xavier 初始化")
        
        # 偏置初始化（通常为0）
        self.biases = np.zeros((1, output_dim))

        # 反向传播需用到的中间变量
        self.input = None
        self.grad_weights = None # 权重梯度
        self.grad_biases = None # 偏置梯度

    def forward(self, x):
        """前向传播：x.shape=(batch_size, input_dim) →output.shape=(batch_size, output_dim)"""
        self.input = x # 保存输入，用于反向传播
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, grad_output):
        """反向传播：计算梯度并返回输入的梯度"""
        # grad_output.shape=(batch_size, output_dim)
        self.grad_weights = np.dot(self.input.T, grad_output) /self.input.shape[0] # 平均梯度（避免批次大小影响）
        self.grad_biases = np.mean(grad_output, axis=0,keepdims=True) # 偏置梯度（按批次平均）
        grad_input = np.dot(grad_output, self.weights.T) # 输入的梯度（传给前一层）
        return grad_input
    def update_params(self, learning_rate):
        """直接用学习率更新参数（适用于SGD）"""
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases