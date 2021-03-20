lang=python
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=../model/$lang
train_file=$data_dir/$lang/TbCS/train-original.jsonl
dev_file=$data_dir/$lang/TbCS/dev-original.jsonl
epochs=40
pretrained_model=microsoft/codebert-base
load_model_path=../model/python/checkpoint-best-bleu/pytorch_model.bin
cuda_devices=0,1

python run.py --do_train --do_eval --model_type roberta --load_model_path $load_model_path --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs --cuda_devices $cuda_devices
