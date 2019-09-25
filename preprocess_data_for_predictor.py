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
import json
import argparse


logger = logging.getLogger(__name__)

def get_batch(data_dir, option, max_seq_length,mode=None,delex=False,no_source=False):
    examples = []
    prev_sys = None
    num = 0

    prediction=None
    if option == 'train':
        with open('{}/train.json'.format(data_dir)) as f:
            source = json.load(f)
    elif option == 'dev':
        with open('{}/val.json'.format(data_dir)) as f:
            source = json.load(f)
    else:
        with open('{}/test.json'.format(data_dir)) as f:
            source = json.load(f)
        with open('/tmp/results.txt.pred.BERT_dim128_w_domain_exp.pred') as f:
            prediction = json.load(f)

    fw = open('data/{}.tsv'.format(option), 'w')
    logger.info("Loading total {} dialogs".format(len(source)))
    for num_dial, dialog_info in enumerate(source):
        hist = []
        hist_segment = []
        dialog_file = dialog_info['file']
        dialog = dialog_info['info']
        sys = "conversation start"
        for turn_num, turn in enumerate(dialog):
            #user = [vocab[w] if w in vocab else vocab['<UNK>'] for w in turn['user'].split()]
            user = turn['user_orig']
            turn_sys=turn['sys_orig']
            if prediction:
                turn_sys=prediction[dialog_file][turn_num]
            if delex:
                user=turn['user']
                turn_sys=turn['sys']
            hierarchical_act_vecs = [0 for _ in range(Constants.act_len)]
            source = []
            for k, v in turn['source'].items():
                source.extend([k.split('_')[1][:-1], 'is', v])
            source = " ".join(source)
            if len(source) == 0 or no_source:
                source = "no information"

            if turn['act'] != "None":
                for w in turn['act']:
                    d, f, s = w.split('-')
                    hierarchical_act_vecs[Constants.domains.index(d)] = 1
                    #for _ in Constants.function_imapping[w]:
                    hierarchical_act_vecs[len(Constants.domains) + Constants.functions.index(f)] = 1                        
                    #for _ in Constants.arguments_imapping[w]:
                    hierarchical_act_vecs[len(Constants.domains) + len(Constants.functions) + Constants.arguments.index(s)] = 1
            if mode=='history':
                sys = sys + '[SEP]' + turn_sys
                print("{}\t{}\t{}\t{}\t{}\t{}".format(dialog_file, str(turn_num), source, sys, user,json.dumps(hierarchical_act_vecs)), file=fw)
                sys = sys + '[SEP]' + user + turn_sys
            elif mode=='truth':
                user = turn_sys
                sys = sys + '[SEP]' + user
                print("{}\t{}\t{}\t{}\t{}\t{}".format(dialog_file, str(turn_num), source, sys, user,json.dumps(hierarchical_act_vecs)), file=fw)
            else:
                print("{}\t{}\t{}\t{}\t{}\t{}".format(dialog_file, str(turn_num), source, sys, user, json.dumps(hierarchical_act_vecs)), file=fw)
                sys = turn_sys


    fw.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--mode",
                        default=None,
                        type=str,
                        help="Input mode. None/history/truth/predict")
    parser.add_argument('--delex',action='store_true')
    parser.add_argument('--no_source',action='store_true')

    args = parser.parse_args()

    get_batch('data/', 'train', 60,args.mode,args.delex,args.no_source)
    get_batch('data/', 'dev', 60,args.mode,args.delex,args.no_source)
    get_batch('data/', 'test', 60,args.mode,args.delex,args.no_source)

