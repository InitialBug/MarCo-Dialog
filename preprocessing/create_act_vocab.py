import json

from transformer import Constants


def token_action(data_dir, option):
    with open('../{}/{}.json'.format(data_dir, option)) as f:
        source = json.load(f)
        f.close()

    for num_dial, dialog_info in enumerate(source):
        dialog_file = dialog_info['file']
        dialog = dialog_info['info']
        for turn_num, turn in enumerate(dialog):
            domain = []
            func = []
            arg = []
            turn['actseq'] = []
            if turn['act']:
                for act in turn['act'].keys():
                    d, f, a = act.split('-')
                    if d not in domain:
                        domain.append(d)
                    if f not in func:
                        func.append(f)
                    if a not in arg:
                        arg.append(a)
            else:
                domain.append('general')
                func.append('none')
                # arg.append('none')
            domain = sorted(domain)
            func = sorted(func)
            # arg=sorted(arg)
            turn['actseq'] = domain + func + arg
    f = open('../{}/{}.json'.format(data_dir, option), 'w')
    json.dump(source, f)


def get_vocab():
    vocab = ["[PAD]", "[EOS]", '[SOS]'] + Constants.domains + Constants.functions + Constants.arguments
    act_vocab = {'rev': {}, 'vocab': {}}
    for i, v in enumerate(vocab):
        act_vocab['rev'][i] = v
        act_vocab['vocab'][v] = i
    with open('../data/act_vocab.json', 'w') as f:
        json.dump(act_vocab, f)


if __name__ == "__main__":
    get_vocab()
    token_action('data', 'train')
    token_action('data', 'val')
    token_action('data', 'test')
