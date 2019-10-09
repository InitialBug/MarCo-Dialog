import json


def seq2vec(option):

    with open('output/act_pred.json') as f:
        precition=json.load(f)

    output=open('output/act_{}_prediction.json'.format(option),'w')
    for file_id,actions in precition.items():
        for i,act in enumerate(actions):
            act=act[:30]+[0]+act[30:]
            actions[i]=act

    json.dump(precition,output,indent=2)

if __name__=='__main__':
    seq2vec('test')
