import json
import torch
import random
import numpy
import logging
import os
import sys
import argparse
import time
from torch.autograd import Variable
from transformer.Transformer import Transformer, ActGenerator,TransformerDecoder
from torch.optim.lr_scheduler import MultiStepLR
import transformer.Constants as Constants
from itertools import chain
from MultiWOZ import get_batch
from transformer.LSTM import LSTMDecoder
from transformer.Semantic_LSTM import SCLSTM
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tools import *
from collections import OrderedDict
from evaluator import evaluateModel
import logging.handlers


logger = logging.getLogger(__name__)
handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename="log")

logger.setLevel(logging.DEBUG)
handler1.setLevel(logging.WARNING)
handler2.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)

logger.addHandler(handler1)
logger.addHandler(handler2)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default="train", help="whether to train or test the model", choices=['train', 'test', 'postprocess'])
    parser.add_argument('--emb_dim', type=int, default=128, help="the embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.2, help="the embedding dimension")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume previous run")
    parser.add_argument('--batch_size', type=int, default=3, help="the embedding dimension")
    parser.add_argument('--model', type=str, default="CNN", help="the embedding dimension")
    parser.add_argument('--data_dir', type=str, default='data', help="the embedding dimension")
    parser.add_argument('--beam_size', type=int, default=2, help="the embedding dimension")
    parser.add_argument('--max_seq_length', type=int, default=100, help="the embedding dimension")
    parser.add_argument('--layer_num', type=int, default=3, help="the embedding dimension")    
    parser.add_argument('--evaluate_every', type=int, default=5, help="the embedding dimension")
    parser.add_argument('--one_hot', default=False, action="store_true", help="whether to use one hot")
    parser.add_argument('--th', type=float, default=0.4, help="the embedding dimension")
    parser.add_argument('--head', type=int, default=4, help="the embedding dimension")
    parser.add_argument("--output_dir", default="checkpoints/generator/", type=str, \
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--output_file", default='output/results.txt.pred', type=str, help="The initial learning rate for Adam.")
    parser.add_argument("--non_delex", default=False, action="store_true", help="The initial learning rate for Adam.")
    parser.add_argument("--hist_num", default=0,type=int, help="The initial learning rate for Adam.")

    args = parser.parse_args()
    return args

args = parse_opt()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

with open("{}/vocab.json".format(args.data_dir), 'r') as f:
    vocabulary = json.load(f)

act_ontology = Constants.act_ontology

vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
tokenizer = Tokenizer(vocab, ivocab, False)

with open("{}/act_vocab.json".format(args.data_dir), 'r') as f:
    act_vocabulary = json.load(f)

act_vocab, act_ivocab = act_vocabulary['vocab'], act_vocabulary['rev']
act_tokenizer = Tokenizer(act_vocab, act_ivocab, False)

logger.info("Loading Vocabulary of {} size".format(tokenizer.vocab_len))
# Loading the dataset

os.makedirs(args.output_dir, exist_ok=True)
checkpoint_file = args.model

if 'train' in args.option:
    *train_examples, _ = get_batch(args.data_dir, 'train', tokenizer, act_tokenizer, args.max_seq_length)
    train_data = TensorDataset(*train_examples)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    *val_examples, val_id = get_batch(args.data_dir, 'test', tokenizer, act_tokenizer, args.max_seq_length)
    dialogs = json.load(open('{}/test.json'.format(args.data_dir)))
    gt_turns = json.load(open('{}/test_reference.json'.format(args.data_dir)))
elif 'test' in args.option or 'postprocess' in args.option:
    *val_examples, val_id = get_batch(args.data_dir, 'test', tokenizer, act_tokenizer, args.max_seq_length)
    dialogs = json.load(open('{}/test.json'.format(args.data_dir)))
    if args.non_delex:
        gt_turns = json.load(open('{}/test_reference_nondelex.json'.format(args.data_dir)))
    else:
        gt_turns = json.load(open('{}/test_reference.json'.format(args.data_dir)))

eval_data = TensorDataset(*val_examples)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

BLEU_calc = BLEUScorer()
F1_calc = F1Scorer()


decoder = ActGenerator(vocab_size=tokenizer.vocab_len,act_vocab_size=act_tokenizer.vocab_len, d_word_vec=args.emb_dim, act_dim=Constants.act_len,
                                 n_layers=args.layer_num, d_model=args.emb_dim, n_head=args.head, dropout=args.dropout)

resp_generator = TransformerDecoder(vocab_size=tokenizer.vocab_len, d_word_vec=args.emb_dim, act_dim=Constants.act_len,
                                 n_layers=args.layer_num, d_model=args.emb_dim, n_head=args.head, dropout=args.dropout)

decoder.to(device)
resp_generator.to(device)
loss_func = torch.nn.BCEWithLogitsLoss()
loss_func.to(device)


ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=Constants.PAD)
ce_loss_func.to(device)

label_list=Constants.functions+Constants.arguments
if args.option == 'train':
    decoder.train()
    resp_generator.train()
    if args.resume:
        decoder.load_state_dict(torch.load(checkpoint_file))
        logger.info("Reloaing the encoder and decoder from {}".format(checkpoint_file))

    logger.info("Start Training with {} batches".format(len(train_dataloader)))

    # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(decoder.parameters())+list(resp_generator.parameters())), betas=(0.9, 0.98), eps=1e-09)
    optimizer2 = torch.optim.Adam(filter(lambda x: x.requires_grad, resp_generator.parameters()), betas=(0.9, 0.98), eps=1e-09)
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, decoder.parameters()), betas=(0.9, 0.98), eps=1e-09)


    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)
    
    best_BLEU = 0
    alpha=0.1
    for epoch in range(360):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, act_vecs, query_results, \
            rep_in, resp_out, belief_state, pred_hierachical_act_vecs, act_user_ids, act_in, act_out, all_label, *_ = batch

            decoder.zero_grad()
            resp_generator.zero_grad()
            logits,act_logits = decoder(tgt_seq=act_in, src_seq=act_user_ids,bs=belief_state)

            loss1 = ce_loss_func(logits.contiguous().view(logits.size(0) * logits.size(1), -1).contiguous(), \
                                act_out.contiguous().view(-1))
            loss2=loss_func(act_logits.view(-1),all_label.view(-1))
            loss=loss1+alpha*loss2
            loss.backward()
            optimizer.step()

            resp_logits = resp_generator(tgt_seq=rep_in, src_seq=input_ids, act_vecs=pred_hierachical_act_vecs,bs=belief_state)

            loss3 = ce_loss_func(resp_logits.contiguous().view(resp_logits.size(0) * resp_logits.size(1), -1).contiguous(), \
                                resp_out.contiguous().view(-1))
            loss3.backward()
            optimizer2.step()

            # if step % 100 == 0:
            #     print("epoch {} step {} training loss {} loss1 {} loss2 {}".format(epoch, step, loss.item(),loss1.item(),loss2.item()))
            #     logger.info("epoch {} step {} training loss {} loss1 {} loss2 {}".format(epoch, step, loss.item(),loss1.item(),loss2.item()))
        alpha=min(1,alpha+0.1*epoch)
        scheduler.step()
        if loss3.item() < 3.0 and loss1.item()<3.0 and epoch > 0 and epoch % args.evaluate_every == 0:
            logger.info("start evaluating BLEU on validation set")
            decoder.eval()
            # Start Evaluating after each epoch
            model_turns = {}
            TP, TN, FN, FP = 0, 0, 0, 0
            for batch_step, batch in enumerate(eval_dataloader):
                all_pred = []
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, act_vecs, query_results, \
                rep_in, resp_out, belief_state, pred_hierachical_act_vecs, act_user_ids, act_in, act_out, all_label, *_ = batch

                hyps,act_logits = decoder.translate_batch(domain=act_in[:,0],bs=belief_state,act_vecs=act_vecs, \
                                               src_seq=act_user_ids, n_bm=args.beam_size,
                                               max_token_seq_len=Constants.ACT_MAX_LEN)
                cls_pred=torch.sigmoid(act_logits)
                for hyp_step, hyp in enumerate(hyps):
                    # pred = tokenizer.convert_id_to_tokens(hyp)
                    # file_name = val_id[batch_step * args.batch_size + hyp_step]
                    # if file_name not in model_turns:
                    #     model_turns[file_name] = [pred]
                    # else:
                    #     model_turns[file_name].append(pred)

                    pre1=[0]*Constants.act_len
                    # pre2=[0]*Constants.act_len

                    for w in hyp:
                        if w not in [Constants.PAD, Constants.EOS]:
                            pre1[w-3] =1
                    # for w in act_out[hyp_step]:
                    #     if w ==Constants.EOS:
                    #         break
                    #     else:
                    #         pre2[w-3] =1

                    all_pred.append(pre1)
                # all_pred=torch.Tensor(all_pred)+cls_pred
                # all_pred=(all_pred > 0.8).long()
                all_pred=torch.Tensor(all_pred)
                all_label=all_label.cpu()
                TP, TN, FN, FP = obtain_TP_TN_FN_FP(all_pred, all_label, TP, TN, FN, FP)

                resp_hyps = resp_generator.translate_batch(bs=belief_state,act_vecs=pred_hierachical_act_vecs, \
                                               src_seq=input_ids, n_bm=args.beam_size,
                                               max_token_seq_len=40)

                for hyp_step, hyp in enumerate(resp_hyps):
                    pred = tokenizer.convert_id_to_tokens(hyp)
                    file_name = val_id[batch_step * args.batch_size + hyp_step]
                    if file_name not in model_turns:
                        model_turns[file_name] = [pred]
                    else:
                        model_turns[file_name].append(pred)


            precision = TP / (TP + FP + 0.001)
            recall = TP / (TP + FN + 0.001)
            F1 = 2 * precision * recall / (precision + recall + 0.001)
            print("precision is {} recall is {} F1 is {}".format(precision, recall, F1))
            logger.info("precision is {} recall is {} F1 is {}".format(precision, recall, F1))
            if F1 > best_BLEU:
                torch.save(decoder.state_dict(), os.path.join(checkpoint_file,str(F1)))
                best_BLEU = F1
            BLEU = BLEU_calc.score(model_turns, gt_turns)
            inform,request=evaluateModel(model_turns,gt_turns)
            print(inform,request,BLEU)
            logger.info("{} epoch, Validation BLEU {}, inform {}, request {} ".format(epoch, BLEU,inform,request))
            # if BLEU > best_BLEU:
            #     torch.save(decoder.state_dict(), os.path.join(checkpoint_file,str(BLEU)))
            #     best_BLEU = BLEU
            decoder.train()
elif args.option == "test":
    decoder.load_state_dict(torch.load(checkpoint_file))
    logger.info("Loading model from {}".format(checkpoint_file))
    decoder.eval()
    logger.info("Start Testing with {} batches".format(len(eval_dataloader)))

    model_turns = {}
    act_turns = {}
    step = 0
    start_time = time.time()
    TP, TN, FN, FP = 0, 0, 0, 0
    for batch_step, batch in enumerate(eval_dataloader):
        all_pred = []
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, act_vecs, query_results, \
        rep_in, resp_out, belief_state, all_label, act_in, act_out, *_ = batch

        # logits, act_logits = decoder(tgt_seq=act_in, src_seq=input_ids, bs=belief_state)
        hyps, act_logits = decoder.translate_batch(domain=act_in[:, 0], bs=belief_state, act_vecs=act_vecs, \
                                                   src_seq=input_ids, n_bm=args.beam_size,
                                                   max_token_seq_len=Constants.ACT_MAX_LEN)
        for hyp_step, hyp in enumerate(hyps):
            pred = act_tokenizer.convert_id_to_tokens(hyp)
            file_name = val_id[batch_step * args.batch_size + hyp_step]
            if file_name not in model_turns:
                model_turns[file_name] = [pred]
            else:
                model_turns[file_name].append(pred)
            pre1 = [0] * Constants.act_len
        # pre2=[0]*Constants.act_len

            for w in hyp:
                if w not in [Constants.PAD, Constants.EOS]:
                    pre1[w - 3] = 1
            # for w in act_out[hyp_step]:
            #     if w ==Constants.EOS:
            #         break
            #     else:
            #         pre2[w-3] =1

            all_pred.append(pre1)
        all_pred=torch.Tensor(all_pred)
        all_label = all_label.cpu()
        # all_pred=(cls_pred > 0.5).long()
        TP, TN, FN, FP = obtain_TP_TN_FN_FP(all_pred, all_label, TP, TN, FN, FP)
        precision = TP / (TP + FP + 0.001)
        recall = TP / (TP + FN + 0.001)
    F1 = 2 * precision * recall / (precision + recall + 0.001)
    print("precision is {} recall is {} F1 is {}".format(precision, recall, F1))

    with open(args.output_file, 'w') as fp:
        model_turns = OrderedDict(sorted(model_turns.items()))
        json.dump(model_turns, fp, indent=2)
# elif args.option == "test":
#     decoder.load_state_dict(torch.load(checkpoint_file))
#     logger.info("Loading model from {}".format(checkpoint_file))
#     decoder.eval()
#     logger.info("Start Testing with {} batches".format(len(eval_dataloader)))
#
#     model_turns = {}
#     act_turns = {}
#     step = 0
#     start_time = time.time()
#     TP, TN, FN, FP = 0, 0, 0, 0
#     for batch_step, batch in enumerate(eval_dataloader):
#         batch = tuple(t.to(device) for t in batch)
#         input_ids, input_mask, segment_ids, act_vecs, query_results, \
#             rep_in, resp_out, belief_state, pred_hierachical_act_vecs, *_ = batch
#
#         hyps = decoder.translate_batch(act_vecs=pred_hierachical_act_vecs, src_seq=input_ids,
#                                        n_bm=args.beam_size, max_token_seq_len=40)
#         for hyp_step, hyp in enumerate(hyps):
#             pred = tokenizer.convert_id_to_tokens(hyp)
#             file_name = val_id[batch_step * args.batch_size + hyp_step]
#             if file_name not in model_turns:
#                 model_turns[file_name] = [pred]
#             else:
#                 model_turns[file_name].append(pred)
#
#         logger.info("finished {}/{} used {} sec/per-sent".format(batch_step, len(eval_dataloader), \
#                                                            (time.time() - start_time) / args.batch_size))
#         start_time = time.time()
#
#     with open(args.output_file, 'w') as fp:
#         model_turns = OrderedDict(sorted(model_turns.items()))
#         json.dump(model_turns, fp, indent=2)
#
#     BLEU = BLEU_calc.score(model_turns, gt_turns)
#     entity_F1 = F1_calc.score(model_turns, gt_turns)
#     inform,request=evaluateModel(model_turns)
#     print("Validation BLEU {}, inform {}, request {} ".format(BLEU, inform, request))
#     logger.info("Validation BLEU {}, inform {}, request {} ".format(BLEU, inform, request))
elif args.option == "postprocess":
    with open(args.output_file, 'r') as f:
        model_turns = json.load(f)
        
    evaluateModel(model_turns)
        
    success_rate = nondetokenize(model_turns, dialogs)
    BLEU = BLEU_calc.score(model_turns, gt_turns)
    
    with open('/tmp/results.txt.pred.non_delex', 'w') as f:
        model_turns = OrderedDict(sorted(model_turns.items()))        
        json.dump(model_turns, f, indent=2)
    logger.info("Validation BLEU {}, Success Rate {}".format(BLEU, success_rate))
    
    with open('/tmp/results.txt.non_delex', 'w') as f:
        json.dump(gt_turns, f, indent=2)
else:
    raise ValueError("No such option")
