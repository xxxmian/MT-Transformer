import json
from collections import Counter
import jieba
import nltk
import re
import unicodedata
import xml.etree.ElementTree
import os
from config import *

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
def pre_process():
    f = open(raw_data_path, 'r')
    all = f.readlines()
    b = []
    en = []
    zh = []
    for i in range(len(all)):
        b.append(all[i].strip().split(sep='\t'))
    for i in range(len(b)):
        en.append(b[i][0])
        zh.append(b[i][1])
    with open('english.json','w') as f:
        json.dump(en, f)
    with open('chinese.json','w') as f:
        json.dump(zh, f)


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]
def build_wordmap_zh():
    zh_path = 'chinese.json'

    with open(zh_path, 'r') as f:
        zh = json.load(f)

    word_freq = Counter()

    for sentence in zh:
        seg_list = jieba.cut(sentence.strip())
        # Update word frequency
        word_freq.update(list(seg_list))

    # Create word map
    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    words = word_freq.most_common(output_lang_vocab_size - 4)
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<start>'] = 1
    word_map['<end>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:10])

    with open('data/wordmap_zh.json', 'w') as file:
        json.dump(word_map, file, indent=4)

def build_wordmap_en():
    en_path = 'english.json'

    with open(en_path, 'r') as f:
        sentences = json.load(f)

    word_freq = Counter()

    for sentence in sentences:
        sentence_en = sentence.strip().lower()
        tokens = [normalizeString(s) for s in nltk.word_tokenize(sentence_en)]
        # Update word frequency
        word_freq.update(tokens)

    # Create word map
    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    words = word_freq.most_common(input_lang_vocab_size - 4)
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<start>'] = 1
    word_map['<end>'] = 2
    word_map['<unk>'] = 3
    print(len(word_map))
    print(words[:10])

    with open('data/wordmap_en.json', 'w') as file:
        json.dump(word_map, file, indent=4)
def extract_valid_data():
    valid_translation_path = os.path.join(valid_folder, 'valid.en-zh.en.sgm')
    with open(valid_translation_path, 'r') as f:
        data_en = f.readlines()
    data_en = [line.replace(' & ', ' &amp; ') for line in data_en]
    with open(valid_translation_path, 'w') as f:
        f.writelines(data_en)

    root = xml.etree.ElementTree.parse(valid_translation_path).getroot()
    data_en = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_folder, 'valid.en'), 'w') as out_file:
        out_file.write('\n'.join(data_en) + '\n')

    root = xml.etree.ElementTree.parse(os.path.join(valid_folder, 'valid.en-zh.zh.sgm')).getroot()
    data_zh = [elem.text.strip() for elem in root.iter() if elem.tag == 'seg']
    with open(os.path.join(valid_folder, 'valid.zh'), 'w') as out_file:
        out_file.write('\n'.join(data_zh) + '\n')

def build_samples():
    word_map_zh = json.load(open('data/wordmap_zh.json', 'r'))
    word_map_en = json.load(open('data/wordmap_en.json', 'r'))

    translation_path_en = 'english.json'
    translation_path_zh = 'chinese.json'

    with open(translation_path_en, 'r') as f:
        data_en = json.load(f)

    with open(translation_path_zh, 'r') as f:
        data_zh = json.load(f)

    train_num = int(len(data_en)*0.8)
    valid_num = len(data_en) - train_num

    filename = 'data/samples_train.json'
    samples = []
    for i in range(train_num):
        sentence_en = data_en[i].strip().lower()
        tokens = [normalizeString(s) for s in nltk.word_tokenize(sentence_en)]
        input_en = encode_text(word_map_en, tokens)

        sentence_zh = data_zh[i].strip()
        seg_list = jieba.cut(sentence_zh)
        output_zh = encode_text(word_map_zh, list(seg_list))

        if len(input_en) <= max_len and len(
                output_zh) <= max_len and UNK_token not in input_en and UNK_token not in output_zh:
            samples.append({'input': list(input_en), 'output': list(output_zh)})
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=4)

    filename = 'data/samples_valid.json'
    samples = []
    for i in range(train_num, len(data_en)):
        sentence_en = data_en[i].strip().lower()
        tokens = [normalizeString(s) for s in nltk.word_tokenize(sentence_en)]
        input_en = encode_text(word_map_en, tokens)

        sentence_zh = data_zh[i].strip()
        seg_list = jieba.cut(sentence_zh)
        output_zh = encode_text(word_map_zh, list(seg_list))

        if len(input_en) <= max_len and len(
                output_zh) <= max_len and UNK_token not in input_en and UNK_token not in output_zh:
            samples.append({'input': list(input_en), 'output': list(output_zh)})
    with open(filename, 'w') as f:
        json.dump(samples, f, indent=4)



if __name__ == '__main__':
    pre_process()
    build_wordmap_en()

    build_wordmap_zh()
    #extract_valid_data()
    build_samples()
    '''
    
    
    '''

