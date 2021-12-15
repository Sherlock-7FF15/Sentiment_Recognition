#!/bin/bash
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -p gpu 
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
module load anaconda/3.7.4
module load nvidia/cuda/10.0 
export PYTHONUNBUFFERED=1
source activate python36
export BERT_BASE_DIR=../bert-models/chinese-bert_chinese_wwm_L-12_H-768_A-12
export NCOV_DIR=../dataset_bert/
export OUTPUT_DIR=../output

CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
  --task_name=COLA \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$NCOV_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR
