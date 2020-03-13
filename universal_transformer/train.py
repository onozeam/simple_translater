import argparse
import os
import pandas as pd
import pickle
import re
import spacy
import torchtext
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import UniversalTransformer


def generate_mask(src, tgt, src_field, tgt_field, cl_args):
    src_pad_idx = src_field.vocab.stoi['<pad>']
    tgt_pad_idx = tgt_field.vocab.stoi['<pad>']
    src_padding_mask = src.T == src_pad_idx
    tgt_padding_mask = tgt.T == tgt_pad_idx
    memory_padding_mask = None
    src_peaking_mask = None
    _mask = torch.tril(torch.ones(tgt.size(0), tgt.size(0))) == 0
    tgt_peaking_mask = torch.zeros(tgt.size(0), tgt.size(0)).masked_fill(_mask, float('-inf'))
    memory_peaking_mask = None
    masks = [src_padding_mask, tgt_padding_mask, memory_padding_mask, src_peaking_mask, tgt_peaking_mask, memory_peaking_mask]
    to_cuda = lambda x: x.cuda() if x is not None else None  # noqa: E731
    if cl_args.device == 'cuda':
        masks = [to_cuda(mask) for mask in masks]
    return masks


def batch_size_fn(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.tgt) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class MyIterator(torchtext.data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in torchtext.data.batch(d, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class Tokenize:
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def __call__(self, data):
        def trim(data):
            data = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(data))
            data = re.sub(r"[ ]+", " ", data)
            data = re.sub(r"\!+", "!", data)
            data = re.sub(r"\,+", ",", data)
            data = re.sub(r"\?+", "?", data)
            data = data.lower()
            return data
        data = trim(data)
        return [token.text for token in self.nlp.tokenizer(data) if token.text != " "]


def _train(model, dataset_iter, optimizer, lr_scheduler, cl_args, src_field, tgt_field, iteration_num):
    print('start trainning...')
    model.train()
    tgt_pad_idx = tgt_field.vocab.stoi['<pad>']
    total_loss = 0
    for epoch in range(cl_args.epochs):
        for iter_, mini_batch in enumerate(dataset_iter):
            src = mini_batch.src
            tgt = mini_batch.tgt
            tgt_input = tgt[:-1, :]
            tgt_label = tgt[1:, :]
            if cl_args.device == 'cuda':
                src, tgt_input, tgt_label = src.cuda(), tgt_input.cuda(), tgt_label.cuda()
            src_padding_mask, tgt_padding_mask, memory_padding_mask, src_peaking_mask, tgt_peaking_mask, memory_peaking_mask =\
                generate_mask(src, tgt_input, src_field, tgt_field, cl_args)
            out, avg_n_updates, avg_remainders = model(src, tgt_input, tgt_mask=tgt_peaking_mask, src_key_padding_mask=src_padding_mask, tgt_key_padding_mask=tgt_padding_mask)  # noqa: E501
            optimizer.zero_grad()
            del src, tgt_input, src_padding_mask, tgt_padding_mask, memory_padding_mask, src_peaking_mask, tgt_peaking_mask, memory_peaking_mask
            loss = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)(out.view(-1, out.size(-1)), tgt_label.view(-1))
            loss += 0.001*(avg_n_updates + avg_remainders)
            del out, tgt_label
            loss.backward()
            optimizer.step()
            if cl_args.lr_scheduling:
                lr_scheduler.step()
            total_loss += loss.item()
            if iter_ != 0 and iter_ % 100 == 0:
                print(f'epoch: {epoch}, iter: {iter_}, loss: {total_loss/100}')
                total_loss = 0
                if iter_ % 300 == 0:
                    print('saving weights...')
                    torch.save(model.state_dict(), f'{cl_args.save_path}/model_state')
                    print('done.')
    print('end trainning.')


def main():
    # initialize variable
    parser = argparse.ArgumentParser(description='Initialize training parameter.')
    parser.add_argument('-device', required=True, type=str, help='"cuda" or "cpu"')
    parser.add_argument('-save_path', type=str, default='saved')
    parser.add_argument('-use_saved_fields', action='store_true')
    parser.add_argument('-use_saved_weights', action='store_true')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=3000)
    parser.add_argument('-max_seq_len', type=int, default=80)
    parser.add_argument('-max_pondering_time', type=int, default=10)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-learning_rate', type=float, default=0.0001)
    parser.add_argument('-nhead', type=int, default=2)
    parser.add_argument('-embedding_dim', type=int, default=512)
    parser.add_argument('-feedforward_dim', type=int, default=2048)
    parser.add_argument('-lr_scheduling', action='store_true')
    args = parser.parse_args()
    src_lang = 'en'
    tgt_lang = 'fr'
    # create train iterator (create field, dataset, iterator)
    # # create field
    if args.use_saved_fields:
        if args.device == 'cpu':
            print('loading saved fields...')
            with open(f'{args.save_path}/src.pickle', 'rb') as s:
                src_field = pickle.load(s)
            with open(f'{args.save_path}/tgt.pickle', 'rb') as t:
                tgt_field = pickle.load(t)
        else:
            exit('use_saved_fields option can be used on only cpu.')
    else:
        print('creating fields...')
        src_field: torchtext.data.field.Field = torchtext.data.Field(lower=True, tokenize=Tokenize(src_lang))
        tgt_field: torchtext.data.field.Field = torchtext.data.Field(lower=True, tokenize=Tokenize(tgt_lang), init_token='<sos>', eos_token='<eos>')
    # # create dataset
    src_data = open("data/english.txt").read().strip().split('\n')
    tgt_data = open("data/french.txt").read().strip().split('\n')
    df = pd.DataFrame({'src': src_data, 'tgt': tgt_data}, columns=["src", "tgt"])
    too_long_mask = (df['src'].str.count(' ') < args.max_seq_len) & (df['tgt'].str.count(' ') < args.max_seq_len)
    df = df.loc[too_long_mask]  # remove too long sentence
    df.to_csv("tmp_dataset.csv", index=False)
    dataset = torchtext.data.TabularDataset('./tmp_dataset.csv', format='csv', fields=[('src', src_field), ('tgt', tgt_field)])
    os.remove('tmp_dataset.csv')
    # # create itrerator
    dataset_iter = MyIterator(dataset, batch_size=args.batch_size, device=args.device,
                              repeat=False, sort_key=lambda x: (len(x.src), len(x.tgt)),
                              batch_size_fn=batch_size_fn, train=True, shuffle=True)
    # build vocab, save field object and add variable.
    src_field.build_vocab(dataset)
    tgt_field.build_vocab(dataset)
    if not args.use_saved_fields:
        print('saving fields...')
        pickle.dump(src_field, open(f'{args.save_path}/src.pickle', 'wb'))
        pickle.dump(tgt_field, open(f'{args.save_path}/tgt.pickle', 'wb'))
    iteration_num = [i for i, _ in enumerate(dataset_iter)][-1]
    # initialize model
    model = UniversalTransformer(n_src_vocab=len(src_field.vocab), n_tgt_vocab=len(tgt_field.vocab),
                                 embedding_dim=args.embedding_dim, nhead=args.nhead, max_seq_len=args.max_seq_len, max_pondering_time=args.max_pondering_time)
    # initialize param
    if args.use_saved_weights:
        print('loading saved model states...')
        model.load_state_dict(torch.load(f'{args.save_path}/model_state'))
    else:
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
    if args.device == 'cuda':
        model = model.cuda()
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = CosineAnnealingLR(optimizer, iteration_num)
    _train(model, dataset_iter, optimizer, lr_scheduler, args, src_field, tgt_field, iteration_num)
    print('saving weights...')
    torch.save(model.state_dict(), f'{args.save_path}/model_state')


if __name__ == '__main__':
    main()
