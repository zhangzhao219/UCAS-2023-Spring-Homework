#!/bin/bash  

MY_PYTHON="/home/coder/miniconda/envs/nlp2/bin/python" # 这里是conda中具体环境下的python绝对路径名称


# 输出路径
OUTPUT_DIR="../output_dir/IEMOCAP/bert-tiny/finetune/run"
# 数据集
data_dir="../../data/dl_data"
dataset_name="IEMOCAP"
# bert-base-uncased  bert-large-uncased
model_name_or_path="prajjwal1/bert-tiny"
cache_dir="../pretrain_models"  # huggingface库预训练模型下载地址


EPOCH=5
BATCH_SIZE=16
val_batch_size=16
LR=1e-4
warmup_ratio=0.1

step_log=25

do_valid=True
do_predict=True
max_debug_samples=0  
max_seq_length=512

EXEC=/home/coder/projects/finetune/bert/run.py # 这里配置实际运行的python文件名

export CUDA_VISIBLE_DEVICES="1"

$MY_PYTHON $EXEC \
--data_dir $data_dir \
--dataset_name $dataset_name \
--epoch $EPOCH \
--output_dir $OUTPUT_DIR \
--batch_size $BATCH_SIZE \
--lr $LR \
--model_name_or_path $model_name_or_path \
--cache_dir $cache_dir \
--val_batch_size $val_batch_size \
--do_valid $do_valid \
--do_predict $do_predict \
--max_debug_samples $max_debug_samples \
--step_log $step_log \
--warmup_ratio $warmup_ratio \
--max_seq_length $max_seq_length \