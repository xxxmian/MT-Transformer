import torch
from model import get_model
from data_gen import read_data, create_masks, nopeak_mask
from config import *
import torch.nn.functional as F
import math
import nltk.translate.bleu_score as bleu
import rouge


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = (k_ix / torch.tensor(k)).to(torch.int64)
    col = k_ix % k
    
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    
    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores
def validation():
    with open('data/wordmap_zh.json','r') as f:
        zh_map = json.load(f)
    valid_data = read_data('valid')
    input_lang = Lang('data/wordmap_en.json')
    output_lang = Lang('data/wordmap_zh.json')
    
    model = get_model(input_lang.n_words, output_lang.n_words)
    model.load_state_dict(torch.load('params.pkl'))
    for i, (src, lengths, trg, max_target_len) in enumerate(valid_data):
        if i==3:
            break
        src = src.transpose(0, 1).to(device)
        trg = trg.transpose(0, 1).to(device)
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input)
        e_output = model.encoder(src, src_mask).to(device)
        
        out = model.out(model.decoder(trg_input,
                                      e_output, src_mask,trg_mask))
        out = F.softmax(out, dim=-1)
        probs, ix = out[:, -1].data.topk(10)
        log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
        outputs = torch.zeros(3, 80).long().to(device)
        outputs[:, 0] = 0
        outputs[:, 1] = ix[0]

        e_outputs = torch.zeros(3, e_output.size(-2), e_output.size(-1)).to(device)
        e_outputs[:, :] = e_output[0]
        
        
        # stage 2
        #outputs, log_scores = k_best_outputs(outputs, out, log_scores, 3, 3)
        length = 5
        print(length)
        return ' '.join([zh_map[tok] for tok in idx[0][1:length]])
        
        '''
        eos_tok = 3
        src_mask = (src != 0).unsqueeze(-2)
        ind = None
        for i in range(0, 80,10):
    
            trg_mask = nopeak_mask(i).to(device)
            #_,trg_mask = create_masks(src, outputs[:,:i])
            out = model.out(model.decoder(outputs[:, :i],
                                          e_outputs, src_mask, trg_mask))
    
            out = F.softmax(out, dim=-1)
    
            outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, 3)
    
            if (outputs == eos_tok).nonzero().size(0) == 3:
                alpha = 0.7
                div = 1 / ((outputs == eos_tok).nonzero()[:, 1].type_as(log_scores) ** alpha)
                _, ind = torch.max(log_scores * div, 1)
                ind = ind.data[0]
                break
        if ind is None:
            length = (outputs[0] == eos_tok).nonzero()[0]
            return ' '.join([zh_map[tok] for tok in outputs[0][1:length]])

        else:
            length = (outputs[ind] == eos_tok).nonzero()[0]
            return ' '.join([zh_map[tok] for tok in outputs[ind][1:length]])
        
        '''
    
    
    
    '''
    hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',\
                   'obeys', 'the', 'commands', 'of', 'the', 'party']
    hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops',\
                   'forever', 'hearing', 'the', 'activity', 'guidebook','that', 'party', 'direct']
    reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',\
                  'ensures', 'that', 'the', 'military', 'will', 'forever','heed', 'Party', 'commands']
    reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which','guarantees', 'the', 'military',\
                  'forces', 'always','being', 'under', 'the', 'command', 'of', 'the','Party']
    reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the','army', 'always',\
                       'to', 'heed', 'the', 'directions','of', 'the', 'party']
    bleu_score = bleu.sentence_bleu([reference1,reference2,reference3],hypothesis1)
    
    elevator = rouge.Rouge()
    hy1 = ' '.join(hypothesis1)
    hy2 = ' '.join(hypothesis2)
    re1 = ' '.join(reference1)
    re2 = ' '.join(reference2)
    rouge_score = elevator.get_scores([hy1],[re1])
    for k in rouge_score:
        for i in k.items():
            print(i)
    #print(rouge_score)
    '''
validation()