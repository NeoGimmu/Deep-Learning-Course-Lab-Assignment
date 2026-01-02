import torch
import sentencepiece as spm
import os
from vocab import SimpleVocab

class BPETokenizer:
    """BPE分词器封装"""
    def __init__(self, vocab_path, src_language='en', tgt_language='fr'):
        self.vocab = torch.load(vocab_path, weights_only=False)
        self.src_language = src_language
        self.tgt_language = tgt_language
        
        # 获取特殊标记
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        # 获取特殊标记的索引
        self.unk_index = self.vocab[src_language][self.unk_token]
        self.pad_index = self.vocab[src_language][self.pad_token]
        self.bos_index = self.vocab[src_language][self.bos_token]
        self.eos_index = self.vocab[src_language][self.eos_token]
    
    def encode(self, sentence, language='en', max_length=None, add_special_tokens=True):
        """将句子编码为token IDs"""
        tokens = self.vocab[language]._tokenizer(sentence)
        
        if max_length is not None:
            if add_special_tokens:
                max_length -= 2  # 留空间给BOS和EOS
            tokens = tokens[:max_length]
        
        token_ids = self.vocab[language](tokens)
        
        if add_special_tokens:
            token_ids = [self.bos_index] + token_ids + [self.eos_index]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids, language='fr', skip_special_tokens=True):
        """将token IDs解码为句子"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids 
                        if token_id not in [self.unk_index, self.pad_index, self.bos_index, self.eos_index]]
        
        # 获取词汇表的反向映射
        vocab_list = self.vocab[language].get_itos()
        tokens = [vocab_list[token_id] for token_id in token_ids]
        
        return ' '.join(tokens)
    
    def get_vocab_size(self, language='en'):
        """获取词汇表大小"""
        return len(self.vocab[language])

class SentencePieceTokenizer:
    """SentencePiece分词器封装"""
    def __init__(self, src_model_path, tgt_model_path):
        # 加载SentencePiece模型
        self.src_sp = spm.SentencePieceProcessor(model_file=src_model_path)
        self.tgt_sp = spm.SentencePieceProcessor(model_file=tgt_model_path)
        
        # 获取特殊标记
        self.unk_token = '<unk>'
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        
        # 获取特殊标记的索引
        self.unk_index = self.src_sp.piece_to_id(self.unk_token)
        self.pad_index = self.src_sp.piece_to_id(self.pad_token)
        self.bos_index = self.src_sp.piece_to_id(self.bos_token)
        self.eos_index = self.src_sp.piece_to_id(self.eos_token)
    
    def encode(self, sentence, language='en', max_length=None, add_special_tokens=True):
        """将句子编码为token IDs"""
        sp_model = self.src_sp if language == 'en' else self.tgt_sp
        
        if add_special_tokens:
            token_ids = sp_model.encode(sentence, out_type=int, add_bos=True, add_eos=True)
        else:
            token_ids = sp_model.encode(sentence, out_type=int, add_bos=False, add_eos=False)
        
        if max_length is not None:
            token_ids = token_ids[:max_length]
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode(self, token_ids, language='fr', skip_special_tokens=True):
        """将token IDs解码为句子"""
        sp_model = self.src_sp if language == 'en' else self.tgt_sp
        
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids 
                        if token_id not in [self.unk_index, self.pad_index, self.bos_index, self.eos_index]]
        
        return sp_model.decode(token_ids)
    
    def get_vocab_size(self, language='en'):
        """获取词汇表大小"""
        sp_model = self.src_sp if language == 'en' else self.tgt_sp
        return sp_model.get_piece_size()

class TokenizerFactory:
    """分词器工厂类"""
    @staticmethod
    def get_tokenizer(tokenizer_type, **kwargs):
        """获取分词器实例"""
        if tokenizer_type == 'bpe':
            vocab_path = kwargs.get('vocab_path', 'tokenizers/bpe_vocabs.pt')
            return BPETokenizer(vocab_path)
        elif tokenizer_type == 'sentencepiece':
            src_model_path = kwargs.get('src_model_path', 'tokenizers/en_spm.model')
            tgt_model_path = kwargs.get('tgt_model_path', 'tokenizers/fr_spm.model')
            return SentencePieceTokenizer(src_model_path, tgt_model_path)
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

# 辅助函数
def collate_fn(batch, pad_index):
    """数据批处理函数"""
    def pad_sequence(sequences, padding_value):
        lengths = [len(seq) for seq in sequences]
        max_length = max(lengths)
        padded_sequences = []
        
        for seq in sequences:
            padding = [padding_value] * (max_length - len(seq))
            padded_sequences.append(torch.cat([seq, torch.tensor(padding, dtype=seq.dtype)]))
        
        return torch.stack(padded_sequences), torch.tensor(lengths)
    
    src_batch, tgt_batch = zip(*batch)
    
    # 对源语言和目标语言序列进行填充
    src_padded, src_lengths = pad_sequence(src_batch, padding_value=pad_index)
    tgt_padded, tgt_lengths = pad_sequence(tgt_batch, padding_value=pad_index)
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths

# 测试代码
if __name__ == '__main__':
    # 测试BPE分词器
    print("Testing BPE Tokenizer...")
    try:
        if os.path.exists('tokenizers/bpe_vocabs.pt'):
            bpe_tokenizer = TokenizerFactory.get_tokenizer('bpe')
            test_sentence = "Hello, how are you?"
            encoded = bpe_tokenizer.encode(test_sentence)
            decoded = bpe_tokenizer.decode(encoded)
            print(f"Original: {test_sentence}")
            print(f"Encoded: {encoded}")
            print(f"Decoded: {decoded}")
    except Exception as e:
        print(f"BPE Tokenizer test failed: {e}")
    
    # 测试SentencePiece分词器
    print("\nTesting SentencePiece Tokenizer...")
    try:
        if os.path.exists('tokenizers/en_spm.model') and os.path.exists('tokenizers/fr_spm.model'):
            sp_tokenizer = TokenizerFactory.get_tokenizer('sentencepiece')
            test_sentence = "Hello, how are you?"
            encoded = sp_tokenizer.encode(test_sentence)
            decoded = sp_tokenizer.decode(encoded)
            print(f"Original: {test_sentence}")
            print(f"Encoded: {encoded}")
            print(f"Decoded: {decoded}")
    except Exception as e:
        print(f"SentencePiece Tokenizer test failed: {e}")
