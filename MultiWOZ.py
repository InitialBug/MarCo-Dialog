from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import time
import json
import numpy as np
import torch
import logging
from transformer import Constants
import copy

logger = logging.getLogger(__name__)

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police', 'bus', 'booking', 'general']
functions = ['inform', 'request', 'recommend', 'book', 'select', 'sorry', 'none']
arguments = ['pricerange', 'id', 'address', 'postcode', 'type', 'food', 'phone', 'name', 'area', 'choice',
             'price', 'time', 'reference', 'none', 'parking', 'stars', 'internet', 'day', 'arriveby', 'departure',
             'destination', 'leaveat', 'duration', 'trainid', 'people', 'department', 'stay']

def get_batch(data_dir, option, tokenizer, act_tokenizer, max_seq_length):
    examples = []
    prev_sys = None
    num = 0

    if option == 'train':
        with open('{}/train.json'.format(data_dir)) as f:
            source = json.load(f)

    elif option == 'val':
        with open('{}/val.json'.format(data_dir)) as f:
            source = json.load(f)
        with open('{}/BERT_dev_prediction.json'.format(data_dir)) as f:
            predicted_acts = json.load(f)
    else:
        with open('{}/test.json'.format(data_dir)) as f:
            source = json.load(f)
        with open('{}/BERT_test_prediction.json'.format(data_dir)) as f:
            predicted_acts = json.load(f)

    logger.info("Loading total {} dialogs".format(len(source)))
    for num_dial, dialog_info in enumerate(source):
        hist = []
        hist_segment = []
        dialog_file = dialog_info['file']
        dialog = dialog_info['info']
        for turn_num, turn in enumerate(dialog):
            # user = [vocab[w] if w in vocab else vocab['<UNK>'] for w in turn['user'].split()]
            user=tokenizer.tokenize(turn['user'])
            tokens = tokenizer.tokenize(turn['user'])
            query = copy.copy(tokens)

            # if 'book' in tokens or 'booked' in tokens or 'booking' in tokens:
            segment_user = 1  # turn_num * 2 if turn_num * 2 < Constants.MAX_SEGMENT else Constants.MAX_SEGMENT - 1
            segment_sys = 2  # turn_num * 2 + 1 if turn_num * 2 + 1 < Constants.MAX_SEGMENT else Constants.MAX_SEGMENT - 1



            if len(hist) == 0:
                if len(tokens) > max_seq_length:
                    tokens = tokens[:max_seq_length]
                segment_ids = [segment_user] * len(tokens)
            else:
                # segment_ids = [0] * (len(hist) + 1) + [1] * len(tokens)
                segment_ids = hist_segment + [Constants.PAD] + [segment_user] * len(tokens)
                tokens = hist + [Constants.SEP_WORD] + tokens

            if len(tokens) > max_seq_length:
                tokens = tokens[-max_seq_length:]

            resp_inp_len = len(tokens)
            source = []
            for k, v in turn['source'].items():
                source.append(k.split('_')[1][:-1])
            act_inp_len = len(user + source)

            hist_tokens=tokens
            tokens += source
            if len(tokens) > (max_seq_length * 2):
                tokens = tokens[-(max_seq_length * 2):]

            resp_input_mask=[0]*resp_inp_len+[1]*(len(tokens)-resp_inp_len)
            act_input_mask=[1]*(len(tokens)-act_inp_len)+[0]*act_inp_len

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            resp_input_mask += [1] * (max_seq_length * 2 - len(input_ids))
            act_input_mask += [1] * (max_seq_length * 2 - len(input_ids))

            input_ids+=[Constants.PAD]*(max_seq_length*2-len(input_ids))



            resp = [Constants.SOS_WORD] + tokenizer.tokenize(turn['sys']) + [Constants.EOS_WORD]

            if len(resp) > Constants.RESP_MAX_LEN:
                resp = resp[:Constants.RESP_MAX_LEN - 1] + [Constants.EOS_WORD]
            else:
                resp = resp + [Constants.PAD_WORD] * (Constants.RESP_MAX_LEN - len(resp))

            resp_inp_ids = tokenizer.convert_tokens_to_ids(resp[:-1])
            resp_out_ids = tokenizer.convert_tokens_to_ids(resp[1:])

            bs = [0] * len(Constants.belief_state)
            if turn['BS'] != "None":
                for domain in turn['BS']:
                    for key, value in turn['BS'][domain]:
                        bs[Constants.belief_state.index(domain + '-' + key)] = 1



            act_vecs = [0] * len(Constants.act_ontology)
            if turn['act'] != "None":
                for w in turn['act']:
                    act_vecs[Constants.act_ontology.index(w)] = 1

            bert_act_seq=[]
            if predicted_acts is not None:
                bert_act_vecs = np.asarray(predicted_acts[dialog_file][str(turn_num)], 'int64')
                domain = []
                func = []
                arg = []
                for i in range(len(bert_act_vecs)):
                    if bert_act_vecs[i]>0:

                        if i<len(domains):
                            d=domains[i]
                            if d not in domain:
                                domain.append(d)
                        else:
                            i-=len(domains)
                            if i <len(functions):
                                f=functions[i]
                                if f not in func:
                                    func.append(f)
                            else:
                                i -= len(functions)
                                a=arguments[i]
                                if a not in arg:
                                    arg.append(a)
                domain = sorted(domain)
                func = sorted(func)
                # arg=sorted(arg)
                bert_act_seq=domain+func+arg


            if len(bert_act_seq) < Constants.ACT_MAX_LEN:
                bert_action_masks = [0] * len(bert_act_seq)
            else:
                bert_action_masks = [0] * (Constants.ACT_MAX_LEN - 1)
            bert_act_seq = [Constants.SOS_WORD] + bert_act_seq + [Constants.EOS_WORD]
            if len(bert_act_seq) > Constants.ACT_MAX_LEN:
                bert_act_seq = bert_act_seq[:Constants.ACT_MAX_LEN - 1] + [Constants.EOS_WORD]
            else:
                bert_act_seq = bert_act_seq + [Constants.PAD_WORD] * (Constants.ACT_MAX_LEN - len(bert_act_seq))
            bert_action_masks += [1] * (Constants.ACT_MAX_LEN - len(bert_action_masks) - 1)
            bert_act_seq = act_tokenizer.convert_tokens_to_ids(bert_act_seq[1:])



                # -----------------------------------------act preprocess----------------------------------------------------

            action = [Constants.SOS_WORD] + turn['actseq'] + [Constants.EOS_WORD]
            if len(turn['actseq'])<Constants.ACT_MAX_LEN:
                action_masks=[0]*len(turn['actseq'])
            else:
                action_masks=[0]*(Constants.ACT_MAX_LEN-1)
            if len(action) > Constants.ACT_MAX_LEN:
                action = action[:Constants.ACT_MAX_LEN - 1] + [Constants.EOS_WORD]
            else:
                action = action + [Constants.PAD_WORD] * (Constants.ACT_MAX_LEN - len(action))
            action_masks+=[1]*(Constants.ACT_MAX_LEN - len(action_masks)-1)
            action_inp_ids = act_tokenizer.convert_tokens_to_ids(action[:-1])
            action_out_ids = act_tokenizer.convert_tokens_to_ids(action[1:])

            labels = [0] * Constants.act_len
            if turn['act']:
                for w in turn['act']:
                    acts = w.split('-')
                    acts = act_tokenizer.convert_tokens_to_ids(acts)
                    labels[acts[0] - 3] = 1

                    labels[acts[1] - 3] = 1

                    labels[acts[2] - 3] = 1
            else:
                acts = ['general', 'none']
                acts = act_tokenizer.convert_tokens_to_ids(acts)
                labels[acts[0] - 3] = 1

                labels[acts[1] - 3] = 1

            examples.append([input_ids, action_masks,resp_inp_ids, resp_out_ids, bs, bert_act_seq,
                             action_inp_ids, action_out_ids, labels, act_input_mask,resp_input_mask,dialog_file])
            num += 1


            sys = tokenizer.tokenize(turn['sys'])
            if turn_num == 0:
                hist = hist_tokens + [Constants.SEP_WORD] + sys
                hist_segment = segment_ids[1:-1] + [Constants.PAD] + [segment_sys] * len(sys)
            else:
                hist = hist + [Constants.SEP_WORD] + hist_tokens + [Constants.SEP_WORD] + sys
                hist_segment = hist_segment + [Constants.PAD] + segment_ids[1:-1] + [Constants.PAD] + [
                    segment_sys] * len(sys)

    all_input_ids = torch.tensor([f[0] for f in examples], dtype=torch.long)
    action_masks = torch.tensor([f[1] for f in examples], dtype=torch.float32).byte()
    all_response_in = torch.tensor([f[2] for f in examples], dtype=torch.long)
    all_response_out = torch.tensor([f[3] for f in examples], dtype=torch.long)
    all_belief_state = torch.tensor([f[4] for f in examples], dtype=torch.float32)
    bert_act_seq = torch.tensor([f[5] for f in examples], dtype=torch.long)

    action_inp_ids = torch.tensor([f[6] for f in examples], dtype=torch.long)
    action_out_ids = torch.tensor([f[7] for f in examples], dtype=torch.long)
    labels = torch.tensor([f[8] for f in examples], dtype=torch.float32)

    act_input_mask = torch.tensor([f[9] for f in examples], dtype=torch.float32).byte()
    resp_input_mask = torch.tensor([f[10] for f in examples], dtype=torch.float32).byte()

    all_files = [f[11] for f in examples]
    # all_template_ids = torch.tensor([f[9] for f in examples], dtype=torch.long)

    return all_input_ids, action_masks, \
            all_response_in, all_response_out, all_belief_state, \
           bert_act_seq, action_inp_ids, action_out_ids, labels,act_input_mask ,resp_input_mask,all_files