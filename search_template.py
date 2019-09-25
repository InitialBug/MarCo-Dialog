
import json

def search(data_dir, option,mode='act'):
    templates = {}

    with open('{}/train.json'.format(data_dir)) as f:
        template_source = json.load(f)

    if option == 'train':
        with open('{}/train.json'.format(data_dir)) as f:
            source = json.load(f)
    elif option == 'dev':
        with open('{}/val.json'.format(data_dir)) as f:
            source = json.load(f)
    else:
        with open('{}/test.json'.format(data_dir)) as f:
            source = json.load(f)

    if mode=='act':
        for  dialog_info in template_source:

            dialog = dialog_info['info']
            dialog_id = dialog_info['file']

            for turn_num, turn in enumerate(dialog):
                action=[]
                domain=None
                if turn['act'] :
                    for w in turn['act']:
                        d, f, s = w.split('-')
                        domain=d
                        # action.append(s)
                        action.append(w)

                    action=sorted(action)
                    action.append(domain)
                    action="_".join(action)
                    if action in templates.keys():
                        if templates[action].count('[')<turn['sys'].count('['):
                            templates[action]=turn['sys']
                    else:
                        templates[action] = turn['sys']

    results={}
    fw = open('data/{}_template.json'.format(option), 'w')
    total=0
    no_act=0
    no_match=0
    for num_dial, dialog_info in enumerate(source):

        dialog_id = dialog_info['file']
        dialog = dialog_info['info']
        results[dialog_id]=[]
        for turn_num, turn in enumerate(dialog):
            action = []
            domain = None
            if turn['act'] :
                for w in turn['act']:
                    d, f, s = w.split('-')
                    domain = d
                    # action.append(s)
                    action.append(w)

                action = sorted(action)
                action.append(domain)
                action = "_".join(action)
                if action in templates.keys():
                    results[dialog_id].append(templates[action])
                else:
                    no_match+=1
                    max_match=-1
                    sys_response=''
                    for tp_action,response in templates.items():
                        count = 0
                        tp_action=tp_action.split('_')
                        if domain == tp_action[-1]:
                            for value in action[:-1]:
                                if value in tp_action:
                                    count+=1
                            if count>max_match:
                                max_match=count
                                sys_response=response
                    results[dialog_id].append(sys_response)
            else:
                no_act+=1
                for k,v in turn['source']:
                    domain=k.split('_')[0]
                    break
                for tp_action, response in templates.items():
                    tp_action = tp_action.split('_')
                    if domain == tp_action[-1]:
                        results[dialog_id].append(response)
                        break
            total+=1
    print('no action:{}, not exact match:{}'.format(no_act/total,no_match/total))
    json.dump(results,fw)
    fw.close()

if __name__=='__main__':
    search('data/', 'test')