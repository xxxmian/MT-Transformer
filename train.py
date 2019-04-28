
import time
import torch
from model import get_model
from data_gen import read_data, create_masks
from config import Lang, learning_rate, epochs, device

import torch.nn.functional as F

def train(model):
    model.train()
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        train_data = read_data('train')

        for i, (src, lengths, trg, max_target_len) in enumerate(train_data):
            src = src.transpose(0, 1).to(device)
            trg = trg.transpose(0, 1).to(device)
            trg_input = trg[:, :-1]

            src_mask, trg_mask = create_masks(src, trg_input)
            pred = model(src, trg_input, src_mask, trg_mask)
            ys = trg[:,1:].contiguous().view(-1)
            optimizer.zero_grad()
            loss = F.cross_entropy(pred.view(-1, pred.size(-1)), ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()

            if (i + 1) % 1 == 0:
                p = int(100 * (i + 1) / len(train_data))
                avg_loss = total_loss / 100

                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, avg_loss), end='\r')

                total_loss = 0

        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100,
               avg_loss, epoch + 1, avg_loss))

def main():
    input_lang = Lang('data/wordmap_en.json')
    output_lang = Lang('data/wordmap_zh.json')
    print("input_lang.n_words: " + str(input_lang.n_words))
    print("output_lang.n_words: " + str(output_lang.n_words))
    model = get_model(input_lang.n_words, output_lang.n_words)
    train(model)
if __name__ == '__main__':
    main()








