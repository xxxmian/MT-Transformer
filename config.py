import json
import os

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = .5
learning_rate = 0.0001
n_iteration = 4000
print_every = 100
save_every = 1
workers = 1
max_len = 10  # Maximum sentence length to consider
min_word_freq = 20  # Minimum word count threshold for trimming
save_dir = 'models'
input_lang_vocab_size = 5000
output_lang_vocab_size = 5000

# Configure models
model_name = 'cb_model'
attn_model = 'general'
start_epoch = 0
epochs = 200
hidden_size = 100
n_layers = 6
heads = 10
dropout = 0.05
batch_size = 10
train_split = 0.9

raw_data_path = 'data/cmn.txt'
train_folder = 'data/train'
valid_folder = 'data/validation'
test_folder = 'data/test'



train_en_filename = 'train.en'
train_zh_filename = 'train.zh'
valid_en_filename = 'valid.en'
valid_zh_filename = 'valid.zh'

src_data_path = os.path.join(train_folder, train_en_filename)
trg_data_path = os.path.join(train_folder, train_zh_filename)
# num_train_samples = 8206380
# num_valid_samples = 7034

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3

start_word = '<start>'
stop_word = '<end>'
unknown_word = '<unk>'


class Lang:
    def __init__(self, filename):
        word_map = json.load(open(filename, 'r'))
        self.word2index = word_map
        self.index2word = {v: k for k, v in word_map.items()}
        self.n_words = len(word_map)



