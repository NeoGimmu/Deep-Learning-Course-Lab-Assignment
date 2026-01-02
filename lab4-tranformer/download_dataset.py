import os
import torch
from torchtext.datasets import TranslationDataset
from torchtext.data import Field
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import sentencepiece as spm
from vocab import SimpleVocab

# 设置参数
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'
MAX_LENGTH = 100
MIN_FREQ = 1
VOCAB_SIZE = 64

# 创建保存目录
os.makedirs('data', exist_ok=True)
os.makedirs('tokenizers', exist_ok=True)

# 获取分词器
def get_tokenizers():
    tokenizers = {}
    # 使用空格分词器替代spaCy，避免依赖语言模型
    tokenizers[SRC_LANGUAGE] = get_tokenizer('basic_english')
    tokenizers[TGT_LANGUAGE] = get_tokenizer('basic_english')  # 对法语也使用基本英文分词器
    return tokenizers

# 构建词汇表
def build_vocabs(train_iter, tokenizers):
    vocabs = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        def yield_tokens(data_iter):
            for data_sample in data_iter:
                yield tokenizers[ln](data_sample[ln])
        vocabs[ln] = build_vocab_from_iterator(
            yield_tokens(train_iter),
            min_freq=MIN_FREQ,
            specials=['<unk>', '<pad>', '<bos>', '<eos>'],
            special_first=True
        )
        vocabs[ln].set_default_index(vocabs[ln]['<unk>'])
    return vocabs

# 数据预处理
def preprocess_sentence(sentence, tokenizer, vocab, max_length=MAX_LENGTH):
    tokens = tokenizer(sentence)
    tokens = tokens[:max_length-2]  # 留空间给 <bos> 和 <eos>
    tokens = ['<bos>'] + tokens + ['<eos>']
    token_ids = vocab(tokens)
    return torch.tensor(token_ids, dtype=torch.long)

# 保存数据
def save_data(data_iter, tokenizers, vocabs, split_name):
    src_data = []
    tgt_data = []
    
    for src, tgt in data_iter:
        src_ids = preprocess_sentence(src, tokenizers[SRC_LANGUAGE], vocabs[SRC_LANGUAGE])
        tgt_ids = preprocess_sentence(tgt, tokenizers[TGT_LANGUAGE], vocabs[TGT_LANGUAGE])
        src_data.append(src_ids)
        tgt_data.append(tgt_ids)
    
    torch.save((src_data, tgt_data), f'data/{split_name}.pt')
    print(f'Saved {split_name} data to data/{split_name}.pt')

# 生成SentencePiece训练数据
def generate_spm_data(data_iter, split_name):
    en_file = f'data/{split_name}.en.txt'
    fr_file = f'data/{split_name}.fr.txt'
    
    with open(en_file, 'w', encoding='utf-8') as f_en, open(fr_file, 'w', encoding='utf-8') as f_fr:
        for src, tgt in data_iter:
            f_en.write(src + '\n')
            f_fr.write(tgt + '\n')
    
    return en_file, fr_file

# 训练SentencePiece模型
def train_sentencepiece(input_file, model_prefix, vocab_size=VOCAB_SIZE):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='unigram',  # 使用unigram模型
        bos_id=0,
        eos_id=1,
        pad_id=2,
        unk_id=3
    )
    print(f'Trained SentencePiece model: {model_prefix}.model')

# 主函数
def main():
    print('Preparing synthetic dataset for testing...')
    
    # 创建少量合成数据用于测试
    synthetic_data = [
        ("Hello world", "Bonjour le monde"),
        ("How are you", "Comment allez-vous"),
        ("I love Python", "J'aime Python"),
        ("Transformer is powerful", "Transformer est puissant"),
        ("Mamba is faster", "Mamba est plus rapide"),
        ("Positional encoding is important", "L'encodage positionnel est important"),
        ("SentencePiece is a good tokenizer", "SentencePiece est un bon tokenizer"),
        ("RoPE is rotary positional encoding", "RoPE est un encodage positionnel rotatif"),
        ("Machine translation is useful", "La traduction automatique est utile"),
        ("Deep learning models are amazing", "Les modèles d'apprentissage profond sont incroyables")
    ]
    
    # 生成SentencePiece训练数据
    print('Generating SentencePiece training data...')
    train_en_file = 'data/train.en.txt'
    train_fr_file = 'data/train.fr.txt'
    
    with open(train_en_file, 'w', encoding='utf-8') as f_en, open(train_fr_file, 'w', encoding='utf-8') as f_fr:
        for en, fr in synthetic_data:
            f_en.write(en + '\n')
            f_fr.write(fr + '\n')
    
    # 训练SentencePiece模型
    print('Training SentencePiece models...')
    train_sentencepiece(train_en_file, 'tokenizers/en_spm')
    train_sentencepiece(train_fr_file, 'tokenizers/fr_spm')
    
    # 构建BPE分词器和词汇表
    print('Building BPE tokenizers and vocabularies...')
    tokenizers = get_tokenizers()
    
    # 创建简单的词汇表
    vocabs = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # 收集所有标记
        all_tokens = []
        for data_sample in synthetic_data:
            sentence = data_sample[0] if ln == SRC_LANGUAGE else data_sample[1]
            all_tokens.extend(tokenizers[ln](sentence))
        
        # 创建词汇表
        vocab = {}
        # 添加特殊标记
        for special in ['<unk>', '<pad>', '<bos>', '<eos>']:
            vocab[special] = len(vocab)
        
        # 添加普通标记
        for token in all_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        # 创建词汇表对象
        vocabs[ln] = SimpleVocab(vocab, vocab['<unk>'])
    
    # 保存BPE词汇表
    torch.save(vocabs, 'tokenizers/bpe_vocabs.pt')
    print('Saved BPE vocabularies to tokenizers/bpe_vocabs.pt')
    
    # 保存预处理后的数据
    print('Preprocessing and saving data...')
    save_data(synthetic_data, tokenizers, vocabs, 'train')
    save_data(synthetic_data, tokenizers, vocabs, 'valid')
    save_data(synthetic_data, tokenizers, vocabs, 'test')
    
    print('Dataset preparation completed with synthetic data!')

if __name__ == '__main__':
    main()
