
import torch
import torch.utils.data as data
import numpy as np
import itertools
import json
from config import batch_size, PAD_token, device
from torch.autograd import Variable
# Returns padded input sequence tensor and lengths


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.izip_longest(*l, fillvalue=fillvalue))
def binaryMatrix(l):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m
def inputVar(indexes_batch):
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(indexes_batch):
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, max_target_len


def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask


def create_masks(src, trg):
    src_mask = (src != PAD_token).unsqueeze(-2).to(device)

    if trg is not None:
        trg_mask = (trg != PAD_token).unsqueeze(-2).to(device)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size).to(device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask



def batch2TrainData(pair_batch):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch)
    output, max_target_len = outputVar(output_batch)

    return inp, lengths, output,  max_target_len


class TranslationDataset(data.Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.num_batches = len(self.samples) // batch_size
        print('count: ' + str(len(self.samples)))

    def __getitem__(self, i):
        start_idx = i * batch_size
        pair_batch = []

        for i_batch in range(batch_size):
            sample = self.samples[start_idx + i_batch]
            pair_batch.append((sample['input'], sample['output']))

        return batch2TrainData(pair_batch)

    def __len__(self):
        return self.num_batches

def read_data(split):
    """
    if src_data_path is not None:
        try:
            src_data = open(src_data_path).read().strip().split('\n')
        except:
            print("error: '" + src_data_path + "' file not found")
            quit()

    if trg_data_path is not None:
        try:
            trg_data = open(trg_data_path).read().strip().split('\n')
        except:
            print("error: '" + trg_data_path + "' file not found")
            quit()
    """
    samples_path = 'data/samples_{}.json'.format(split)
    samples = json.load(open(samples_path, 'r'))
    _data = TranslationDataset(samples)
    return _data
if __name__ == '__main__':
    a=read_data('train')
    for i in range(10):
        temp = a[i]