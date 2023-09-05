# nugpt

This repo includes pertinent codebase for developing a miniaturized GPT model. It follows GPT-2 architecture. 

# Commands

## Data Engineering

* Include text data
* run: `python data/data/prepare.py`

## Model Training

* `python train.py --batch_size=32 --wandb_log=True`

## Inference

* `python sample.py --out_dir=out-wiki`