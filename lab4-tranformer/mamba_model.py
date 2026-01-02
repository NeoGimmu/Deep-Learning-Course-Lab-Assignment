import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSMLayer(nn.Module):
    """Selective State Space Model 层"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        # 输入投影
        self.in_proj = nn.Linear(d_model, d_model * expand)
        
        # 因果卷积
        self.conv1d = nn.Conv1d(
            in_channels=d_model * expand,
            out_channels=d_model * expand,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model * expand
        )
        
        # 门控和状态投影
        self.x_proj = nn.Linear(d_model * expand, d_state * 2)
        self.state_proj = nn.Linear(d_model * expand, d_state)
        
        # 输出投影
        self.out_proj = nn.Linear(d_state, d_model)
        
        # 初始化卷积层
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        if self.conv1d.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1d.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.conv1d.bias, -bound, bound)
    
    def selective_scan(self, x, state=None):
        batch_size, seq_len, d_in = x.shape
        device = x.device
        
        # 计算A、B参数
        # x_proj 应该是 [batch_size, seq_len, 2*d_state]
        x_proj = self.x_proj(x)
        A = -torch.exp(x_proj[..., :self.d_state])  # (B, L, D_state)
        B = F.silu(x_proj[..., self.d_state:])     # (B, L, D_state)
        
        # 初始化状态
        if state is None:
            state = torch.zeros(batch_size, self.d_state, device=device)
        
        # 选择性扫描
        outputs = []
        for t in range(seq_len):
            # 扩展B和x到合适的维度进行乘法
            # B_t: [batch_size, d_state]
            # x_t: [batch_size, d_in]
            B_t = B[:, t, :]  # [batch_size, d_state]
            x_t = x[:, t, :]  # [batch_size, d_in]
            
            # 计算状态更新
            # 我们需要将 B_t 和 x_t 进行有效的组合
            # 一种简单的方法是将 B_t 扩展为 [batch_size, d_state, 1]，x_t 扩展为 [batch_size, 1, d_in]
            # 然后相乘并求和得到 [batch_size, d_state]
            state_update = (B_t.unsqueeze(2) * x_t.unsqueeze(1)).sum(dim=2)
            
            # 更新状态
            state = state * torch.exp(A[:, t, :]) + state_update
            
            # 保存当前状态作为输出
            outputs.append(state)
        
        # 将输出列表转换为张量 [batch_size, seq_len, d_state]
        output = torch.stack(outputs, dim=1)
        
        return output, state
    
    def forward(self, x, state=None):
        batch_size, seq_len, d_model = x.shape
        
        # 输入投影
        x = self.in_proj(x)
        
        # 因果卷积
        x = x.transpose(1, 2)  # (B, C, L)
        x = self.conv1d(x)[:, :, :seq_len]  # 去除填充
        x = x.transpose(1, 2)  # (B, L, C)
        
        # 激活函数
        x = F.silu(x)
        
        # 选择性扫描
        ssm_output, new_state = self.selective_scan(x, state)
        
        # 输出投影
        # ssm_output 现在是 [batch_size, seq_len, d_state]
        # 需要将其投影回 [batch_size, seq_len, d_model]
        x = self.out_proj(ssm_output)
        
        return x, new_state

class MambaBlock(nn.Module):
    """Mamba 块"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 层归一化
        self.ln = nn.LayerNorm(d_model)
        
        # SSM层
        self.ssm = SSMLayer(d_model, d_state, d_conv, expand)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, state=None):
        residual = x
        x = self.ln(x)
        x, new_state = self.ssm(x, state)
        x = self.dropout(x)
        x = x + residual
        return x, new_state

class MambaEncoder(nn.Module):
    """Mamba 编码器"""
    def __init__(self, vocab_size, d_model=512, num_layers=6, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba块
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.ln_final = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, state=None):
        batch_size, seq_len = x.shape
        
        # 嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        # Mamba层
        if state is None:
            state = [None] * len(self.layers)
        
        new_states = []
        for layer, s in zip(self.layers, state):
            x, new_state = layer(x, s)
            new_states.append(new_state)
        
        # 最终层归一化
        x = self.ln_final(x)
        
        return x, new_states

class MambaDecoder(nn.Module):
    """Mamba 解码器"""
    def __init__(self, vocab_size, d_model=512, num_layers=6, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba块
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.ln_final = nn.LayerNorm(d_model)
        
        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, state=None):
        batch_size, seq_len = x.shape
        
        # 嵌入
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        # Mamba层
        if state is None:
            state = [None] * len(self.layers)
        
        new_states = []
        for layer, s in zip(self.layers, state):
            x, new_state = layer(x, s)
            new_states.append(new_state)
        
        # 最终层归一化
        x = self.ln_final(x)
        
        # 输出层
        x = self.fc_out(x)
        
        return x, new_states

class MambaModel(nn.Module):
    """完整的Mamba模型（用于机器翻译）"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_encoder_layers=6, num_decoder_layers=6,
                 d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        
        # 编码器
        self.encoder = MambaEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_encoder_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # 解码器
        self.decoder = MambaDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器前向传播
        enc_output, enc_state = self.encoder(src)
        
        # 解码器前向传播
        dec_output, dec_state = self.decoder(tgt, enc_output)
        
        return dec_output

# 创建Mamba模型的辅助函数
def create_mamba_model(src_vocab_size, tgt_vocab_size, d_model=512):
    """创建Mamba模型"""
    model = MambaModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    )
    return model
