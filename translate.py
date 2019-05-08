
import time
import torch
from model import get_model
from beam import beam_search
import torch.nn.functional as F


import pdb

import argparse
from config import *
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
from data_gen import read_data, create_masks


def get_synonym(word):
    with open('data/wordmap_en.json','r') as f:
        map_en = json.load(f)
    if word in map_en:
        print(word,'is founded',map_en[word])
    syns = map_en.get(word)
    return syns
    



def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt):
    model.eval()
    indexed = []
    toks = sentence.split(' ')
    for tok in toks:
        indexed.append(get_synonym(tok))
    indexed.append(get_synonym(' . '))
    sentence = torch.IntTensor(indexed)
    if opt.device == 0:
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, opt)
    
    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)


def translate(opt, model):
    sentences = opt.text.lower().split('.')
    print('sen',sentences)
    translated = []
    
    for sentence in sentences:
        translated.append(translate_sentence(sentence , model, opt).capitalize())
    
    return (' '.join(translated))


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=6)
    #parser.add_argument('-src_lang', required=True)
    #parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-heads', type=int, default=10)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    
    opt = parser.parse_args()
    
    opt.device = 0 if opt.no_cuda is False else -1
    
    assert opt.k > 0
    assert opt.max_len > 10
    input_lang = Lang('data/wordmap_en.json')
    output_lang = Lang('data/wordmap_zh.json')
    print("input_lang.n_words: " + str(input_lang.n_words))
    print("output_lang.n_words: " + str(output_lang.n_words))
    model = get_model(input_lang.n_words, output_lang.n_words)
    model.load_state_dict(torch.load('params.pkl'))

    valid_data = read_data('valid')
    model.eval()
    for i, (src, lengths, trg, max_target_len) in enumerate(valid_data):
        src = src.transpose(0, 1).to(device)
        trg = trg.transpose(0, 1).to(device)
        trg_input = trg[:, :-1]
    
        src_mask, trg_mask = create_masks(src, trg_input)
        pred = model(src, trg_input, src_mask, trg_mask)
        ys = trg[:, 1:].contiguous().view(-1)
        
        
       

    while True:
        opt.text = 'I hope so.Call me.'
        '''
        opt.text = input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text == "q":
            break
        if opt.text == 'f':
            fpath = input("Enter a file name to translate \n")
            try:
                opt.text = ' '.join(open(fpath, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        '''
        phrase = translate(opt, model)
        print('> ' + phrase + '\n')


if __name__ == '__main__':
    main()