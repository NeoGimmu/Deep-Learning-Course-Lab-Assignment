import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """原始Transformer的位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class RoPE(nn.Module):
    """旋转位置编码 (Rotary Position Embedding)"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 计算频率
        theta = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
        self.register_buffer('theta', theta)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # 生成位置索引
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        
        # 计算旋转角度
        angle = position * self.theta
        
        # 复制角度以匹配特征维度
        cos_angle = torch.cos(angle).repeat_interleave(2, dim=1)
        sin_angle = torch.sin(angle).repeat_interleave(2, dim=1)
        
        # 将输入分为实部和虚部
        x_real = x
        x_imag = torch.zeros_like(x)
        
        # 应用旋转
        rotated_x = torch.empty_like(x)
        rotated_x[:, :, 0::2] = x_real[:, :, 0::2] * cos_angle[:, 0::2] - x_real[:, :, 1::2] * sin_angle[:, 0::2]
        rotated_x[:, :, 1::2] = x_real[:, :, 0::2] * sin_angle[:, 0::2] + x_real[:, :, 1::2] * cos_angle[:, 0::2]
        
        return rotated_x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        
        return output

class FeedForwardNetwork(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, src_mask):
        # 多头注意力
        attn_output = self.mha(x, x, x, src_mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, tgt_mask, src_tgt_mask):
        # 掩码多头注意力
        attn_output1 = self.mha1(x, x, x, tgt_mask)
        attn_output1 = self.dropout1(attn_output1)
        out1 = self.layernorm1(x + attn_output1)
        
        # 编码器-解码器注意力
        attn_output2 = self.mha2(out1, enc_output, enc_output, src_tgt_mask)
        attn_output2 = self.dropout2(attn_output2)
        out2 = self.layernorm2(out1 + attn_output2)
        
        # 前馈网络
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3

class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, d_ff=2048, dropout=0.1, pos_encoding='original'):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        if pos_encoding == 'original':
            self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        elif pos_encoding == 'rope':
            self.positional_encoding = RoPE(d_model)
        else:
            raise ValueError(f"Unknown positional encoding type: {pos_encoding}")
        
        # 编码器和解码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask, tgt_mask, src_tgt_mask):
        # 源语言嵌入和位置编码
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        if self.pos_encoding_type == 'original':
            src_embedded = self.positional_encoding(src_embedded)
        elif self.pos_encoding_type == 'rope':
            src_embedded = self.positional_encoding(src_embedded)
        
        # 目标语言嵌入和位置编码
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        if self.pos_encoding_type == 'original':
            tgt_embedded = self.positional_encoding(tgt_embedded)
        elif self.pos_encoding_type == 'rope':
            tgt_embedded = self.positional_encoding(tgt_embedded)
        
        # 编码器前向传播
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # 解码器前向传播
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask, src_tgt_mask)
        
        # 输出层
        output = self.fc_out(dec_output)
        
        return output

# 创建模型的辅助函数
def create_transformer_model(src_vocab_size, tgt_vocab_size, pos_encoding='original', d_model=512):
    """创建Transformer模型"""
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        pos_encoding=pos_encoding
    )
    return model
