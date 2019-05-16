import torch.nn as nn
import torch
import os
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer import Constants
from transformer.Models import get_non_pad_mask, get_sinusoid_encoding_table, get_attn_key_pad_mask, get_subsequent_mask
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertEmbeddings
from torch.utils import checkpoint
import pdb
from random import random
from transformer.Beam import Beam

CONFIG_NAME = 'bert_config.json'

class BertDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, decoder_config, embedding, device, dropout=0.1):

        super().__init__()
        self.len_max_seq = decoder_config['len_max_seq']
        d_word_vec = decoder_config['d_word_vec']
        n_layers = decoder_config['n_layers']
        n_head = decoder_config['n_head']
        d_k = decoder_config['d_k']
        d_v = decoder_config['d_v']
        d_model = decoder_config['d_model']
        d_inner = decoder_config['d_inner']  # should be equal to d_model
        vocab_size = decoder_config['vocab_size']

        self.device = device
        self.n_position = self.len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.n_position, d_word_vec, padding_idx=0),
            freeze=True)
        
        self.embedding = embedding

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.last_linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_seq, src_seq, enc_output):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)
            
        tgt_pos = torch.arange(1, tgt_seq.size(-1) + 1).unsqueeze(0).repeat(tgt_seq.size(0), 1).to(self.device)
        # -- Forward
        dec_output = self.embedding(tgt_seq) + self.position_enc(tgt_pos)
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
        
        return self.last_linear(dec_output)


class BertAbsSum(nn.Module):
    def __init__(self, bert_model_path, decoder_config, device):
        super().__init__()

        self.bert_encoder = BertModel.from_pretrained(bert_model_path)
        bert_config_file = os.path.join(bert_model_path, CONFIG_NAME)
        bert_config = BertConfig.from_json_file(bert_config_file)
        self.device = device
        self.bert_emb = BertEmbeddings(bert_config)
        self.decoder = BertDecoder(decoder_config, self.bert_emb, device) 
        self.teacher_forcing = 0.5
    
    def forward(self, src, src_mask, tgt, tgt_mask):
        # src/tgt: [batch_size, seq_len]

        # shift right
        tgt = tgt[:, :-1]
        tgt_mask = tgt_mask[:, :-1]
        # bert input: BertModel.forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True)
        # token_type_ids is not important since we only have one sentence so we can use default all zeros
        bert_encoded = self.bert_encoder(src, attention_mask=src_mask, output_all_encoded_layers=False)[0]  # [batch_size, seq_len, hidden_size]
        # transformer input: BertDecoder.forward(self, tgt_seq_embedded, tgt_pos, src_seq, enc_output, return_attns=False)
        logits = self.decoder(tgt, src, bert_encoded)  # [batch_size, seq_len, vocab_size]
        return logits
    
    def beam_decode(self, src_seq, src_mask, beam_size, n_best):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, beam_size):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * beam_size, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, beam_size)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, beam_size):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, src_seq, enc_output, n_active_inst, beam_size):
                dec_output = self.decoder(dec_seq, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = nn.functional.log_softmax(dec_output, dim=1)
                word_prob = word_prob.view(n_active_inst, beam_size, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, src_seq, enc_output, n_active_inst, beam_size)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            #-- Encode
            src_enc = self.bert_encoder(src_seq, attention_mask=src_mask, output_all_encoded_layers=False)[0]

            #-- Repeat data for beam search
            n_inst, len_s, d_h = src_enc.size()
            src_seq = src_seq.repeat(1, beam_size).view(n_inst * beam_size, len_s)
            src_enc = src_enc.repeat(1, beam_size, 1).view(n_inst * beam_size, len_s, d_h)

            #-- Prepare beams
            inst_dec_beams = [Beam(beam_size, device=self.device) for _ in range(n_inst)]

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.decoder.len_max_seq + 1):

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, beam_size)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)

        return batch_hyp, batch_scores

    def greedy_decode(self, src_seq, src_mask):
        enc_output = self.bert_encoder(src_seq, attention_mask=src_mask, output_all_encoded_layers=False)[0]
        dec_seq = torch.full((src_seq.size(0), ), Constants.BOS).unsqueeze(-1).type_as(src_seq)

        for i in range(self.decoder.len_max_seq):
            dec_output = self.decoder(dec_seq, src_seq, enc_output, 1)
            dec_output = dec_output.max(-1)[1]
            dec_seq = torch.cat((dec_seq, dec_output[:, -1].unsqueeze(-1)), 1)
        return dec_seq











        
