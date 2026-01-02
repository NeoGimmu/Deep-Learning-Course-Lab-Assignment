import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import argparse
import os
import json

from transformer_model import create_transformer_model
from mamba_model import create_mamba_model
from tokenizers import TokenizerFactory, collate_fn

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='Transformer and Mamba Training and Testing')
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'mamba'],
                        help='Model type to use')
    parser.add_argument('--pos_encoding', type=str, default='original', choices=['original', 'rope'],
                        help='Positional encoding type (for transformer only)')
    parser.add_argument('--tokenizer', type=str, default='bpe', choices=['bpe', 'sentencepiece'],
                        help='Tokenizer type')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--quantize', type=str, default=None, choices=['int8'],
                        help='Quantization type')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum sequence length')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Interval for logging training progress')
    return parser.parse_args()

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, data_path):
        self.src_data, self.tgt_data = torch.load(data_path)
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

# 获取数据加载器
def get_dataloaders(data_dir, batch_size, pad_index):
    train_dataset = TranslationDataset(os.path.join(data_dir, 'train.pt'))
    valid_dataset = TranslationDataset(os.path.join(data_dir, 'valid.pt'))
    test_dataset = TranslationDataset(os.path.join(data_dir, 'test.pt'))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_index)
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_index)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_index)
    )
    
    return train_loader, valid_loader, test_loader

# 创建模型
def create_model(model_type, pos_encoding, tokenizer, d_model, device):
    # 获取词汇表大小
    src_vocab_size = tokenizer.get_vocab_size(language='en')
    tgt_vocab_size = tokenizer.get_vocab_size(language='fr')
    
    if model_type == 'transformer':
        model = create_transformer_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            pos_encoding=pos_encoding,
            d_model=d_model
        )
    elif model_type == 'mamba':
        model = create_mamba_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)

# 计算模型大小
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)  # MB

# 训练函数
def train(model, train_loader, valid_loader, optimizer, criterion, device, epochs, log_interval, checkpoint_dir):
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    training_times = []
    
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_total = 0
        
        for batch_idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            
            # 准备目标输入和输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码（仅对Transformer模型）
            if hasattr(model, 'encoder_layers'):  # Transformer模型
                # 源语言掩码 [batch_size, 1, 1, src_seq_len]
                src_mask = (src != tokenizer.pad_index).unsqueeze(1).unsqueeze(2)
                
                # 目标语言掩码（因果掩码） [batch_size, 1, tgt_seq_len, tgt_seq_len]
                tgt_seq_len = tgt_input.size(1)
                tgt_pad_mask = (tgt_input != tokenizer.pad_index).unsqueeze(1).unsqueeze(3)
                causal_mask = torch.tril(torch.ones((1, 1, tgt_seq_len, tgt_seq_len), device=device)).bool()
                tgt_mask = tgt_pad_mask & causal_mask
                
                # 源-目标掩码 [batch_size, 1, tgt_seq_len, src_seq_len]
                src_tgt_mask = (src != tokenizer.pad_index).unsqueeze(1).unsqueeze(2)
                src_tgt_mask = src_tgt_mask.repeat(1, 1, tgt_seq_len, 1)
                
                # 前向传播
                optimizer.zero_grad()
                output = model(src, tgt_input, src_mask, tgt_mask, src_tgt_mask)
            else:  # Mamba模型
                # 前向传播
                optimizer.zero_grad()
                output = model(src, tgt_input)
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * src.size(0)
            train_total += src.size(0)
            
            # 记录训练进度
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = train_loss / train_total
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {avg_loss:.4f}')
        
        # 计算训练耗时
        end_time = time.time()
        epoch_time = end_time - start_time
        training_times.append(epoch_time)
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_total
        train_losses.append(avg_train_loss)
        
        # 验证模型
        avg_valid_loss = evaluate(model, valid_loader, criterion, device)
        valid_losses.append(avg_valid_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] Summary:')
        print(f'  Training Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s')
        print(f'  Validation Loss: {avg_valid_loss:.4f}')
        
        # 保存最佳模型
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            model_path = os.path.join(checkpoint_dir, f'best_model_{model_type}_{pos_encoding}_{tokenizer_type}.pt')
            torch.save(model.state_dict(), model_path)
            print(f'  Saved best model to {model_path}')
    
    return {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'training_times': training_times,
        'best_valid_loss': best_valid_loss,
        'total_training_time': sum(training_times)
    }

# 评估函数
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for src, tgt, src_lengths, tgt_lengths in data_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # 准备目标输入和输出
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # 创建掩码（仅对Transformer模型）
            if hasattr(model, 'encoder_layers'):  # Transformer模型
                # 源语言掩码 [batch_size, 1, 1, src_seq_len]
                src_mask = (src != tokenizer.pad_index).unsqueeze(1).unsqueeze(2)
                
                # 目标语言掩码（因果掩码） [batch_size, 1, tgt_seq_len, tgt_seq_len]
                tgt_seq_len = tgt_input.size(1)
                tgt_pad_mask = (tgt_input != tokenizer.pad_index).unsqueeze(1).unsqueeze(3)
                causal_mask = torch.tril(torch.ones((1, 1, tgt_seq_len, tgt_seq_len), device=device)).bool()
                tgt_mask = tgt_pad_mask & causal_mask
                
                # 源-目标掩码 [batch_size, 1, tgt_seq_len, src_seq_len]
                src_tgt_mask = (src != tokenizer.pad_index).unsqueeze(1).unsqueeze(2)
                src_tgt_mask = src_tgt_mask.repeat(1, 1, tgt_seq_len, 1)
                
                # 前向传播
                output = model(src, tgt_input, src_mask, tgt_mask, src_tgt_mask)
            else:  # Mamba模型
                # 前向传播
                output = model(src, tgt_input)
            
            # 计算损失
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            
            total_loss += loss.item() * src.size(0)
            total_samples += src.size(0)
    
    return total_loss / total_samples

# 主函数
def main():
    global model_type, pos_encoding, tokenizer_type, tokenizer
    
    args = parse_args()
    model_type = args.model_type
    pos_encoding = args.pos_encoding
    tokenizer_type = args.tokenizer
    device = args.device
    quantize = args.quantize
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    d_model = args.d_model
    max_length = args.max_length
    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    log_interval = args.log_interval
    
    # 检查CUDA可用性
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # 获取设备
    device = torch.device(device)
    
    # 加载分词器
    print(f"Loading {tokenizer_type} tokenizer...")
    tokenizer = TokenizerFactory.get_tokenizer(tokenizer_type)
    
    # 加载数据
    print(f"Loading data from {data_dir}...")
    train_loader, valid_loader, test_loader = get_dataloaders(
        data_dir, batch_size, tokenizer.pad_index
    )
    
    # 创建模型
    print(f"Creating {model_type} model with {pos_encoding} positional encoding...")
    model = create_model(model_type, pos_encoding, tokenizer, d_model, device)
    
    # 打印模型信息
    print(f"Model size: {get_model_size(model):.2f} MB")
    print(f"Source vocab size: {tokenizer.get_vocab_size('en')}")
    print(f"Target vocab size: {tokenizer.get_vocab_size('fr')}")
    
    # 量化模型（如果需要）
    if quantize == 'int8':
        print("Quantizing model to INT8...")
        model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d},  # 需要量化的层类型
            dtype=torch.qint8
        )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_index)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    print(f"Starting training on {device}...")
    training_results = train(
        model, train_loader, valid_loader, optimizer, criterion, device,
        epochs, log_interval, checkpoint_dir
    )
    
    # 测试模型
    print(f"Testing best model...")
    best_model_path = os.path.join(checkpoint_dir, f'best_model_{model_type}_{pos_encoding}_{tokenizer_type}.pt')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        training_results['test_loss'] = test_loss
    else:
        print(f"Best model not found at {best_model_path}")
    
    # 保存训练结果
    results_path = os.path.join(checkpoint_dir, f'training_results_{model_type}_{pos_encoding}_{tokenizer_type}.json')
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=4)
    
    print(f"Training completed! Results saved to {results_path}")
    
    # 返回结果用于比较
    return {
        'model_type': model_type,
        'pos_encoding': pos_encoding,
        'tokenizer': tokenizer_type,
        'total_training_time': training_results['total_training_time'],
        'test_loss': training_results.get('test_loss', float('inf')),
        'model_size': get_model_size(model)
    }

if __name__ == '__main__':
    main()
