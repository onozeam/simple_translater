import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class UniversalTransformer(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, embedding_dim, nhead, max_seq_len, max_pondering_time):
        super(UniversalTransformer, self).__init__()
        self.src_embedding = nn.Embedding(n_src_vocab, embedding_dim)
        self.tgt_embedding = nn.Embedding(n_tgt_vocab, embedding_dim)
        self.src_positional_encoder = PositionalEncoder(embedding_dim, max_seq_len)
        self.tgt_positional_encoder = PositionalEncoder(embedding_dim, max_seq_len)
        self.src_time_step_encoder = TimeStepEncoder(embedding_dim, max_pondering_time)
        self.tgt_time_step_encoder = TimeStepEncoder(embedding_dim, max_pondering_time)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead)
        self.encoder_norm = nn.Linear(embedding_dim, embedding_dim)
        self.decoder_norm = nn.Linear(embedding_dim, embedding_dim)
        self.out = nn.Linear(embedding_dim, n_tgt_vocab)
        self.max_pondering_time = max_pondering_time

    def forward(self, src, tgt,
                src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory, en_n_updates, en_remainders = self.encoder(src, src_mask, src_key_padding_mask)
        out, de_n_updates, de_remainders = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        avg_n_updates = (en_n_updates.sum() + de_n_updates.sum()) / (en_n_updates.numel() + de_n_updates.numel())
        avg_remainders = (en_remainders.sum() + de_remainders.sum()) / (en_remainders.numel() + de_remainders.numel())
        return self.out(out), avg_n_updates, avg_remainders

    def encoder(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.src_embedding(src)
        memory, en_n_updates, en_remainders = self.dynamic_halting_encoder(src, src_mask, src_key_padding_mask)
        return memory, en_n_updates, en_remainders

    def decoder(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt = self.tgt_embedding(tgt)
        out, de_n_updates, de_remainders = self.dynamic_halting_decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return out, de_n_updates, de_remainders

    def dynamic_halting_encoder(self, src, src_mask=None, src_key_padding_mask=None):   # todo: =None
        stacked_state = torch.zeros_like(src)
        act = AdaptiveComputationTime(src.size(1), src.size(0), src.size(2), self._is_cuda)
        if self._is_cuda:
            act = act.cuda()
        for step in range(self.max_pondering_time):
            update_weights, n_updates, remainders = act(src)
            src = self.src_positional_encoder(src)
            src = self.src_time_step_encoder(src, step)
            src = self.transformer_encoder_layer(src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            stacked_state = src*update_weights + stacked_state*(1 - update_weights)
            if not act.should_continue():
                break
        memory = self.encoder_norm(stacked_state)
        return memory, n_updates, remainders

    def dynamic_halting_decoder(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        stacked_state = torch.zeros_like(tgt)
        act = AdaptiveComputationTime(tgt.size(1), tgt.size(0), tgt.size(2), self._is_cuda)
        if self._is_cuda:
            act = act.cuda()
        for step in range(self.max_pondering_time):
            update_weights, n_updates, remainders = act(tgt)
            tgt = self.tgt_positional_encoder(tgt)
            tgt = self.tgt_time_step_encoder(tgt, step)
            tgt = self.transformer_decoder_layer(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)  # todo: 適当に書いたので間違ってるかも
            stacked_state = tgt*update_weights + stacked_state*(1 - update_weights)
            if not act.should_continue():
                break
        out = self.decoder_norm(stacked_state)
        return out, n_updates, remainders

    @property
    def _is_cuda(self):
        return next(self.parameters()).is_cuda


class AdaptiveComputationTime(nn.Module):
    def __init__(self, batch_size, seq_size, hidden_size, is_cuda):
        super(AdaptiveComputationTime, self).__init__()
        self.pondering = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.halting_probability = torch.zeros(seq_size, batch_size).cuda() if is_cuda else torch.zeros(seq_size, batch_size)
        self.still_running_mask = torch.ones(seq_size, batch_size).cuda() if is_cuda else torch.ones(seq_size, batch_size)
        self.remainders = torch.zeros(seq_size, batch_size).cuda() if is_cuda else torch.zeros(seq_size, batch_size)
        self.n_updates = torch.zeros(seq_size, batch_size).cuda() if is_cuda else torch.zeros(seq_size, batch_size)
        self.threshold = 1.0 - 1e-10

    def forward(self, state):
        """
        Args:
            - state: shape is (S, N, E).
            - step: int
        Returns:
            - update_weights: shape is (S, N, 1)
            - n_updates: shape is (S, N)
            - remainders: shape is (S, N)

        where S is the sequence length, N is the batch size, E is the embedding size.
        """
        p = self.sigmoid(self.pondering(state)).squeeze(-1)  # 今回のhalting_probability
        new_halted_mask = (self.halting_probability + p*self.still_running_mask > self.threshold).float() * self.still_running_mask  # 今回の更新で thresholdを超える場合は1, それ以外は0になる. (今回の更新の前からthresholdを超えていた場合は0)  # noqa: E501
        self.still_running_mask = (self.halting_probability + p*self.still_running_mask <= self.threshold).float() * self.still_running_mask  # 今回の更新を終えても, まだthresholdを超えない場合は1, それ以外は0になる.  # noqa: E501
        self.halting_probability += p*self.still_running_mask   # 今回もthresholdを超えなかった場所の, halting_probabilityを更新
        self.remainders += new_halted_mask * (1 - self.halting_probability)  # remainderを更新
        self.halting_probability += new_halted_mask * self.remainders  # 今回thresholdを超えた場所のhalting_probabilityを更新 (1にする)
        self.n_updates += self.still_running_mask + new_halted_mask  # 今回更新された箇所のn_updatesを更新(1を足す)
        update_weights = (p*self.still_running_mask + self.remainders*new_halted_mask).unsqueeze(-1)  # 今回の更新された箇所のself.halting_probability or remainders (今回更新されていない箇所は0)  # noqa: E501
        return update_weights, self.n_updates, self.remainders

    def should_continue(self):
        return (self.halting_probability < self.threshold).any().item()  # thresholdを超えていないhalting_probabilityが一箇所でもあればTrue


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        enc = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                enc[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        enc = enc.unsqueeze(1)
        self.register_buffer('enc', enc)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(0)
        enc = Variable(self.enc[:seq_len], requires_grad=False)
        if x.is_cuda:
            enc.cuda()
        x = x + enc
        return self.dropout(x)


class TimeStepEncoder(PositionalEncoder):
    def forward(self, x, step):
        x = x * math.sqrt(self.d_model)
        enc = Variable(self.enc[step], requires_grad=False)
        if x.is_cuda:
            enc.cuda()
        x = x + enc
        return self.dropout(x)
