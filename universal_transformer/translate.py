import argparse
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import os
import pandas as pd
from model import UniversalTransformer
import torchtext
from train import Tokenize  # noqa: F401


def create_peaking_mask(seq_idx):
    _mask = torch.tril(torch.ones(seq_idx, seq_idx)) == 0
    tgt_peaking_mask = torch.zeros(seq_idx, seq_idx).masked_fill(_mask, float('-inf'))
    return tgt_peaking_mask


def create_field(max_seq_len, save_path):
    src_lang = 'en'
    tgt_lang = 'fr'
    # create train iterator (create field, dataset, iterator)
    # # create field
    src_field: torchtext.data.field.Field = torchtext.data.Field(lower=True, tokenize=Tokenize(src_lang))
    tgt_field: torchtext.data.field.Field = torchtext.data.Field(lower=True, tokenize=Tokenize(tgt_lang), init_token='<sos>', eos_token='<eos>')
    # # create dataset
    src_data = open("data/english.txt").read().strip().split('\n')
    tgt_data = open("data/french.txt").read().strip().split('\n')
    df = pd.DataFrame({'src': src_data, 'tgt': tgt_data}, columns=["src", "tgt"])
    too_long_mask = (df['src'].str.count(' ') < max_seq_len) & (df['tgt'].str.count(' ') < max_seq_len)
    df = df.loc[too_long_mask]  # remove too long sentence
    df.to_csv("tmp_dataset.csv", index=False)
    dataset = torchtext.data.TabularDataset('./tmp_dataset.csv', format='csv', fields=[('src', src_field), ('tgt', tgt_field)])
    os.remove('tmp_dataset.csv')
    # # create itrerator
    # build vocab, save field object and add variable.
    src_field.build_vocab(dataset)
    tgt_field.build_vocab(dataset)
    print('saving fields...')
    pickle.dump(src_field, open(f'{save_path}/src.pickle', 'wb'))
    pickle.dump(tgt_field, open(f'{save_path}/tgt.pickle', 'wb'))
    return src_field, tgt_field


def latest_beamout_and_score(beam_out, out, stacked_score, seq_idx, beam_size):
    """
    beam_outとscoreの更新.
    Args:
        beam_out: (max_seq_len, beam_size)
        out: (i, beam_size, vocab_size)
        stacked_score: (K)
    """
    last_topk_score, last_topk_idx = out[-1, :].log().data.topk(beam_size)  # (S, K), (S, K)
    tmp_score = (last_topk_score.T + stacked_score).T  # (S, K)
    topk_score, topk_flatten_idx = tmp_score.view(-1).topk(beam_size)  # (K), (K)
    topk_beam_idx = topk_flatten_idx // beam_size  # (K)
    latest_score = stacked_score[topk_beam_idx] + topk_score  # (K)
    last_topk_row = topk_flatten_idx // beam_size
    last_topk_col = topk_flatten_idx % beam_size
    beam_out[:seq_idx] = beam_out[:seq_idx, topk_beam_idx]
    beam_out[seq_idx, :] = last_topk_idx[last_topk_row, last_topk_col]
    return beam_out, latest_score


def beam_search(model, src, src_field, tgt_field, max_seq_len, beam_size):
    # pad: 1, sos: 2, eos: 3
    beam_out, memory, stacked_score = first_beam_search(model, src, src_field, tgt_field, max_seq_len, beam_size)   # (max_seq_len, beam_size), (src_seq_len, beam_size, embed_size), (1, beam_size)  # noqa: E501
    stacked_score = stacked_score.squeeze(0)
    eos_tok = tgt_field.vocab.stoi['<eos>']
    topk_sentence, topk_score = [], []
    finished = 0
    for seq_idx in range(2, max_seq_len):
        tgt_peaking_mask = create_peaking_mask(seq_idx)
        tgt = beam_out[:seq_idx, :]
        out, _, _ = model.decoder(tgt, memory, tgt_mask=tgt_peaking_mask)
        out = model.out(out)
        out = F.softmax(out, dim=-1)  # (i, beam_size, vocab_size)
        beam_out, stacked_score = latest_beamout_and_score(beam_out, out, stacked_score, seq_idx, beam_size)
        for beam_idx in range(beam_size):
            if beam_out[seq_idx, beam_idx] == eos_tok:
                topk_score.append(stacked_score[beam_idx].item() / (seq_idx-1))
                topk_sentence.append(beam_out[:seq_idx, beam_idx].clone())
                stacked_score[beam_idx] = -np.inf
                finished += 1
        if finished == beam_size:
            break
    if finished < beam_size:
        """ eosが出現しないsentenceがあれば, 予測を打ち切ってtop1のみ取り出す """
        score, beam_idx = stacked_score.topk(1)
        topk_score.append(score.item())
        topk_sentence.append(beam_out[:seq_idx, beam_idx.item()].clone())
    best_idx = np.asarray(topk_score).argmax()
    return ' '.join([tgt_field.vocab.itos[tok] for tok in topk_sentence[best_idx][1:]])  # translated_sentence


def first_beam_search(model, src, src_field, tgt_field, max_seq_len, beam_size):
    sos_token = tgt_field.vocab.stoi['<sos>']
    src_key_padding_mask = src == src_field.vocab.stoi['<pad>']
    tgt = torch.LongTensor([[sos_token]])
    tgt_peaking_mask = create_peaking_mask(1)
    memory, _, _ = model.encoder(src, src_key_padding_mask=src_key_padding_mask.T)
    out, _, _ = model.decoder(tgt, memory, tgt_mask=tgt_peaking_mask)
    out = model.out(out)
    output = F.softmax(out, dim=-1)
    vals, idxes = output[:, -1].data.topk(beam_size)
    # stacked_score
    stacked_score = vals.log()
    # output
    outputs = torch.zeros(beam_size, max_seq_len).long()
    outputs[:, 0] = sos_token
    outputs[:, 1] = idxes[0]
    # memory
    memories = torch.zeros(beam_size, memory.size(-2), memory.size(-1))
    memories[:, :] = memory[0]
    return outputs.T, memories.transpose(0, 1), stacked_score


def translate(sentence, device='cpu', save_path='saved', embedding_dim=512, nhead=2, max_seq_len=80, max_pondering_time=10, dropout=0.5, beam_size=3):
    # load field
    if os.path.exists(f'{save_path}/src.pickle') and os.path.exists(f'{save_path}/tgt.pickle'):
        print('loading saved fields...')
        with open(f'{save_path}/src.pickle', 'rb') as s:
            src_field = pickle.load(s)
        with open(f'{save_path}/tgt.pickle', 'rb') as t:
            tgt_field = pickle.load(t)
    else:
        print('creating fields...')
        src_field, tgt_field = create_field(max_seq_len, save_path)
    # load model
    model = UniversalTransformer(n_src_vocab=len(src_field.vocab), n_tgt_vocab=len(tgt_field.vocab),
                                 embedding_dim=embedding_dim, nhead=nhead, max_seq_len=max_seq_len, max_pondering_time=max_pondering_time)
    print('loading weights...')
    if device == 'cpu':
        model.load_state_dict(torch.load(f'{save_path}/model_state', map_location=torch.device('cpu')))
    else:
        raise NotImplementedError('prediction on GPU is not implemented.')
    if device == 'cuda':
        model = model.cuda()
    # setence to torch variable
    # sentence = 'they could not speak English.'
    sentence = src_field.preprocess(sentence)
    indexed = [src_field.vocab.stoi[word] for word in sentence]
    src = torch.autograd.Variable(torch.LongTensor([indexed]))

    model.eval()
    result = beam_search(model, src, src_field, tgt_field, max_seq_len, beam_size)
    return result


def main():
    # initialize variable
    parser = argparse.ArgumentParser(description='Initialize training parameter.')
    parser.add_argument('-sentence', required=True, type=str, help='translation target.')
    parser.add_argument('-device', type=str, default='cpu', help='"cuda" or "cpu"')
    parser.add_argument('-save_path', type=str, default='saved')
    parser.add_argument('-batch_size', type=int, default=3000)
    parser.add_argument('-max_seq_len', type=int, default=80)
    parser.add_argument('-max_pondering_time', type=int, default=10)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-nhead', type=int, default=2)
    parser.add_argument('-embedding_dim', type=int, default=512)
    parser.add_argument('-feedforward_dim', type=int, default=2048)
    parser.add_argument('-beam_size', type=int, default=3)
    args = parser.parse_args()

    result = translate(args.sentence, device=args.device, save_path=args.save_path,
                       embedding_dim=args.embedding_dim, nhead=args.nhead, max_seq_len=args.max_seq_len, max_pondering_time=args.max_pondering_time,
                       dropout=args.dropout, beam_size=args.beam_size)
    print(result)


if __name__ == '__main__':
    main()
