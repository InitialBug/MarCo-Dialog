import argparse
import logging.handlers
import os
import random
from collections import OrderedDict

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from MultiWOZ import get_batch
from evaluator import evaluateModel
from tools import *
from transformer.Transformer import RespGenerator, UncertaintyLoss


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default="train", help="whether to train or test the model", choices=['train', 'test', 'postprocess'])
    parser.add_argument('--emb_dim', type=int, default=128, help="the embedding dimension")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume previous run")
    parser.add_argument('--batch_size', type=int, default=3, help="train/dev/test batch size")
    parser.add_argument('--model', type=str, default="model", help="path to save or load models")
    parser.add_argument('--data_dir', type=str, default='data', help="data dir")
    parser.add_argument('--beam_size', type=int, default=2, help="beam size of act/response generator")
    parser.add_argument('--max_seq_length', type=int, default=50, help="max input length")
    parser.add_argument('--ngram', type=int, default=3, help="avoid n gram repeatness")
    parser.add_argument('--layer_num', type=int, default=3, help="transformer layer num")
    parser.add_argument('--evaluate_every', type=int, default=5, help="checkpoints")
    parser.add_argument('--head', type=int, default=4, help="head num for transformer")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--output_file", default='output', type=str, help="path to save generated act/response")
    parser.add_argument("--non_delex", default=False, action="store_true", help="non delex testing")
    parser.add_argument("--hist_num", default=0,type=int, help="turn num of history")
    parser.add_argument('--log', type=str, default='log', help="log file")

    parser.add_argument('--act_source',  type=str, choices=["pred", "bert",'groundtruth'], default='pred', help="action source for validate/test")
    parser.add_argument('--seed', type=int, default=1, help="random seed for initialization")
    parser.add_argument('--label_smoothing', type=float, default=0.0, help="label smoothing rate")

    args = parser.parse_args()
    return args


args = parse_opt()

if args.option == 'train':
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    args.log = os.path.join(args.model, 'train.log')
elif args.option == 'test':
    dir = os.path.dirname(args.model)
    args.log = os.path.join(dir, 'test.log')

logger = logging.getLogger(__name__)
handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename=args.log)

logger.setLevel(logging.DEBUG)
handler1.setLevel(logging.WARNING)
handler2.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler1.setFormatter(formatter)
handler2.setFormatter(formatter)

logger.addHandler(handler1)
logger.addHandler(handler2)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
setup_seed(args.seed)

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

checkpoint_file = args.model

if 'train' in args.option:
    *train_examples, _ = get_batch(args.data_dir, 'train', tokenizer, act_tokenizer, args.max_seq_length)
    train_data = TensorDataset(*train_examples)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    *val_examples, val_id = get_batch(args.data_dir, 'val', tokenizer, act_tokenizer, args.max_seq_length)
    dialogs = json.load(open('{}/val.json'.format(args.data_dir)))
    gt_turns = json.load(open('{}/val_reference.json'.format(args.data_dir)))
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

best_BLEU = 0

weighted_loss_func = UncertaintyLoss(2)
weighted_loss_func.to(device)

resp_generator = RespGenerator(vocab_size=tokenizer.vocab_len,
                               act_vocab_size=act_tokenizer.vocab_len,
                               d_word_vec=args.emb_dim,
                               act_dim=Constants.act_len,
                               n_layers=args.layer_num,
                               d_model=args.emb_dim,
                               n_head=args.head,
                               dropout=args.dropout)

resp_generator.to(device)

bce_loss_func = torch.nn.BCEWithLogitsLoss()
bce_loss_func.to(device)

ce_loss_func = torch.nn.CrossEntropyLoss(ignore_index=Constants.PAD)
ce_loss_func.to(device)

label_list = Constants.functions + Constants.arguments

if args.option == 'train':
    resp_generator.train()
    if args.resume:
        logger.info("Reloaing the encoder and act_generator from {}".format(checkpoint_file))

    logger.info("Start Training with {} batches".format(len(train_dataloader)))

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(resp_generator.parameters()) + list(weighted_loss_func.parameters())), betas=(0.9, 0.98), eps=1e-09)

    scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150, 200], gamma=0.5)

    alpha = 0.1
    for epoch in range(51):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, action_masks, rep_in, resp_out, belief_state,\
                bert_act_seq, act_in, act_out, all_label, act_input_mask,\
                resp_input_mask, *_ = batch

            resp_generator.zero_grad()

            # act loss
            logits, _, act_vecs = resp_generator.act_forward(
                tgt_seq=act_in, src_seq=input_ids, bs=belief_state, input_mask=act_input_mask)

            loss1 = ce_loss_func(
                logits.contiguous().view(logits.size(0) * logits.size(1), -1).contiguous(),
                act_out.contiguous().view(-1))

            # response loss
            resp_logits = resp_generator.resp_forward(tgt_seq=rep_in, src_seq=input_ids, act_vecs=act_vecs,
                                                      act_mask=action_masks, input_mask=resp_input_mask)
            loss2 = ce_loss_func(
                resp_logits.contiguous().view(resp_logits.size(0) * resp_logits.size(1), -1).contiguous(),
                resp_out.contiguous().view(-1))

            # overall loss
            if epoch < 10:
                loss = loss1
            else:
                loss = weighted_loss_func(loss1, loss2)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("epoch {} \tstep {} training \ttotal_loss {:.6f} \tact_loss {:.6f} \tresp_loss {:.6f}".format(epoch, step, loss.item(), loss1.item(), loss2.item()))

        alpha = min(1, alpha + 0.1 * epoch)
        scheduler.step()

        if loss2.item() < 3.0 and loss1.item() < 3.0 and epoch > 0 and epoch % args.evaluate_every == 0:
            logger.info("start evaluating BLEU on validation set")
            resp_generator.eval()
            # Start Evaluating after each epoch
            model_turns = {}
            TP, TN, FN, FP = 0, 0, 0, 0
            for batch_step, batch in enumerate(eval_dataloader):
                all_pred = []
                batch = tuple(t.to(device) for t in batch)
                input_ids, action_masks, rep_in, resp_out, belief_state, \
                    bert_act_seq, act_in, act_out, all_label, act_input_mask, \
                    resp_input_mask, *_ = batch

                if args.act_source == 'bert':
                    act_in = bert_act_seq

                elif args.act_source == 'pred':
                    hyps, act_logits = resp_generator.act_translate_batch(input_mask=act_input_mask, bs=belief_state, \
                                                                          src_seq=input_ids, n_bm=args.beam_size,
                                                                          max_token_seq_len=Constants.ACT_MAX_LEN)
                    for hyp_step, hyp in enumerate(hyps):
                        pre1 = [0] * Constants.act_len
                        if len(hyp) < Constants.ACT_MAX_LEN:
                            hyps[hyp_step] = list(hyps[hyp_step]) + [Constants.PAD] * (Constants.ACT_MAX_LEN - len(hyp))
                        for w in hyp:
                            if w not in [Constants.PAD, Constants.EOS]:
                                pre1[w - 3] = 1
                        all_pred.append(pre1)
                    all_pred = torch.Tensor(all_pred)
                    all_label = all_label.cpu()

                    TP, TN, FN, FP = obtain_TP_TN_FN_FP(all_pred, all_label, TP, TN, FN, FP)
                    act_in = torch.tensor(hyps, dtype=torch.long).to(device)
                else:
                    pass

                _, _, act_vecs = resp_generator.act_forward(tgt_seq=act_in, src_seq=input_ids, bs=belief_state,
                                                            input_mask=act_input_mask)
                action_masks = act_in.eq(Constants.PAD) + act_in.eq(Constants.EOS)
                resp_hyps = resp_generator.resp_translate_batch(bs=belief_state, act_vecs=act_vecs,
                                                                act_mask=action_masks, input_mask=resp_input_mask,
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
            print("precision is {:.6f} recall is {:.6f} F1 is {:.6f}".format(precision, recall, F1))
            logger.info("precision is {:.6f} recall is {:.6f} F1 is {:.6f}".format(precision, recall, F1))
            BLEU = BLEU_calc.score(model_turns, gt_turns)
            inform, request = evaluateModel(model_turns)
            print("{} epoch, Validation BLEU {:.4f}, inform {:.2f}, request {:.2f}, score {:.2f}".format(epoch, BLEU, inform, request, (inform + request) / 2 + 100 * BLEU))
            logger.info("{} epoch, Validation BLEU {:.4f}, inform {:.2f}, request {:.2f}, score {:.2f}".format(epoch, BLEU, inform, request, (inform + request) / 2 + 100 * BLEU))
            if request > best_BLEU:
                save_name = 'inform-{:.2f}-request-{:.2f}-bleu-{:.4f}-seed-{}'.format(inform, request, BLEU, args.seed)
                torch.save(resp_generator.state_dict(), os.path.join(checkpoint_file, save_name))
                best_BLEU = request
                resp_file = os.path.join(args.output_file, 'resp_pred.json')
                with open(resp_file, 'w') as fp:
                    model_turns = OrderedDict(sorted(model_turns.items()))
                    json.dump(model_turns, fp, indent=2)
            resp_generator.train()

elif args.option == "test":
    resp_generator.load_state_dict(torch.load(args.model))
    logger.info("Loading model from {}".format(checkpoint_file))
    resp_generator.eval()
    # Start Evaluating after each epoch
    model_turns = {}
    act_turns={}
    TP, TN, FN, FP = 0, 0, 0, 0
    example_success={}
    for batch_step, batch in enumerate(eval_dataloader):
        all_pred = []
        batch = tuple(t.to(device) for t in batch)
        input_ids, action_masks, rep_in, resp_out, belief_state, bert_act_seq, act_in, act_out, all_label, \
        act_input_mask, resp_input_mask, *_ = batch

        if args.act_source == 'bert':
            act_in = bert_act_seq
        elif args.act_source == 'pred':
            hyps, act_logits = resp_generator.act_translate_batch(input_mask=act_input_mask, bs=belief_state, \
                                                                  src_seq=input_ids, n_bm=args.beam_size,
                                                                  max_token_seq_len=Constants.ACT_MAX_LEN)
            for hyp_step, hyp in enumerate(hyps):
                pre1 = [0] * Constants.act_len
                for w in hyp:
                    if w not in [Constants.PAD, Constants.EOS]:
                        pre1[w - 3] = 1
                if len(hyp) < Constants.ACT_MAX_LEN:
                    hyps[hyp_step] = list(hyps[hyp_step]) + [Constants.PAD] * (Constants.ACT_MAX_LEN - len(hyp))
                all_pred.append(pre1)
                file_name = val_id[batch_step * args.batch_size + hyp_step]
                if file_name not in act_turns:
                    act_turns[file_name] = [pre1]
                else:
                    act_turns[file_name].append(pre1)

            all_pred=torch.Tensor(all_pred)
            all_label=all_label.cpu()
            TP, TN, FN, FP = obtain_TP_TN_FN_FP(all_pred, all_label, TP, TN, FN, FP)

            act_in = torch.tensor(hyps, dtype=torch.long).to(device)
        else:
            pass
        _, _, act_vecs = resp_generator.act_forward(tgt_seq=act_in, src_seq=input_ids, bs=belief_state,
                                                    input_mask=act_input_mask)
        action_masks = act_in.eq(Constants.PAD) + act_in.eq(Constants.EOS)
        resp_hyps = resp_generator.resp_translate_batch(bs=belief_state, act_vecs=act_vecs, act_mask=action_masks,
                                                        input_mask=resp_input_mask,
                                                        src_seq=input_ids, n_bm=args.beam_size,
                                                        max_token_seq_len=40,gram_num=args.ngram)

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
    print("precision is {:.6f} recall is {:.6f} F1 is {:.6f}".format(precision, recall, F1))
    logger.info("precision is {:.6f} recall is {:.6f} F1 is {:.6f}".format(precision, recall, F1))
    BLEU = BLEU_calc.score(model_turns, gt_turns)
    inform, request = evaluateModel(model_turns, example_success)
    print("Test BLEU {:.4f}, inform {:.2f}, request {:.2f}, score {:.2f}".format(BLEU, inform, request, (inform + request) / 2 + 100 * BLEU))
    logger.info("Test BLEU {:.4f}, inform {:.2f}, request {:.2f}, score {:.2f}".format(BLEU, inform, request, (inform + request) / 2 + 100 * BLEU))

    resp_file = os.path.join(args.output_file, 'resp_pred.json')
    with open(resp_file, 'w') as fp:
        model_turns = OrderedDict(sorted(model_turns.items()))
        json.dump(model_turns, fp, indent=2)

    act_file = os.path.join(args.output_file, 'act_pred.json')
    with open(act_file, 'w') as fp:
        act_turns = OrderedDict(sorted(act_turns.items()))
        json.dump(act_turns, fp, indent=2)

    with open('output/example_statistic.json','w') as f:
        json.dump(example_success,f)

    save_name = 'inform-{:.2f}-request-{:.2f}-bleu-{:.4f}'.format(inform, request, BLEU)
    torch.save(resp_generator.state_dict(), os.path.join('model', save_name))

elif args.option == "postprocess":

    resp_file = os.path.join(args.output_file, 'resp_pred.json')
    with open(resp_file, 'r') as f:
        model_turns = json.load(f)

    success_rate = nondetokenize(model_turns, dialogs)
    BLEU = BLEU_calc.score(model_turns, gt_turns)
    print(BLEU)

    resp_file = os.path.join(args.output_file, 'resp_non_delex_pred.json')
    with open(resp_file, 'w') as fp:
        model_turns = OrderedDict(sorted(model_turns.items()))
        json.dump(model_turns, fp, indent=2)
