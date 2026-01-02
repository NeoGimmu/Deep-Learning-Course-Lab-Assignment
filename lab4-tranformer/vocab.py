import torch

# 简单的词汇表类
class SimpleVocab:
    def __init__(self, vocab_dict, default_index):
        self.vocab = vocab_dict
        self.default_index = default_index
    def __call__(self, tokens):
        return [self.vocab.get(token, self.default_index) for token in tokens]
    def __getitem__(self, token):
        return self.vocab.get(token, self.default_index)
    def set_default_index(self, index):
        self.default_index = index
    def get_default_index(self):
        return self.default_index
    def _tokenizer(self, sentence):
        # 简单的空格分词
        return sentence.split()
    def get_itos(self):
        # 返回索引到字符串的映射
        return {index: token for token, index in self.vocab.items()}
    def __len__(self):
        # 返回词汇表大小
        return len(self.vocab)
