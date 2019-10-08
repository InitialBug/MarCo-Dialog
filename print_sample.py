import json


def restore(option):
    with open('data/{}.json'.format(option)) as f:
        source=json.load(f)
    with open('output/resp_pred.json') as f:
        precition=json.load(f)

    output=open('output/{}_restore.json'.format(option),'w')
    for num_dial, dialog_info in enumerate(source):
        dialog_file = dialog_info['file']
        dialog = dialog_info['info']
        for turn_num, turn in enumerate(dialog):
            response=precition[dialog_file][turn_num]
            turn['pred_sys']=response



    json.dump(source,output,indent=2)

if __name__=='__main__':
    restore('test')
