# MarCo

This is the PyTorch implementation of the paper: Multi-Domain Dialogue Acts and Response Co-Generation.

## Model Architecture
<p align="center">
<img src="resource/marco.png" width="50%" />
</p>

## Response Generator

This module is used to control the language generation based on the output of the pre-trained act predictor. The training data is already preprocessed and put in data/ folder (train.json, val.json and test.json).

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python train_generator.py --option train --model model/ --batch_size 384 --max_seq_length 50 --act_source bert
```

### Delexicalized Testing (The entities are normalzied into placeholder like [restaurant_name])

```bash
CUDA_VISIBLE_DEVICES=0 python train_generator.py --option test --model model/MarCo_BERT --batch_size 384 --max_seq_length 50 --act_source bert
```

### Requirements

```
torch==1.0.1
pytorch_pretrained_bert(option)
```