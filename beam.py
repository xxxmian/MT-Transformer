import torch
import torch.nn.functional as F
import math
import json
from data_gen import nopeak_mask

def init_vars(src, model, opt):
    init_tok = 0
    src_mask = (src != 0).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]])
    if opt.device == 0:
        outputs = outputs.cuda()
    
    trg_mask = nopeak_mask(1)
    
    out = model.out(model.decoder(outputs,
                                  e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long()
    if opt.device == 0:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2), e_output.size(-1))
    if opt.device == 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k
    
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]
    
    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores


def beam_search(src, model, opt):
    with open('data/wordmap_en.json','r') as f:
        en_map = json.load(f)
    with open('data/wordmap_zh.json','r') as f:
        zh_map = json.load(f)
    outputs, e_outputs, log_scores = init_vars(src, model, opt)
    eos_tok = 3
    src_mask = (src != 0).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
        
        trg_mask = nopeak_mask(i)
        
        out = model.out(model.decoder(outputs[:, :i],
                                      e_outputs, src_mask, trg_mask))
        
        out = F.softmax(out, dim=-1)
        
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        
        if (outputs == eos_tok).nonzero().size(0) == opt.k:
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


