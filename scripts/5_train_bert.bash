mkdir -p OUTPUT/LP/BERT/

CUDA_VISIBLE_DEVICES= 0,1,2,3 python -m torch.distributed.launch --master_port=9820 --nproc_per_node=4 finetune_simplified.py \
    --model_type bert \
    --tokenizer_name=bert-base-uncased \
    --model_name_or_path bert-base-uncased \
    --config_name bert-base-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --save_steps -1 \
    --per_gpu_eval_batch_size=1   \
    --per_gpu_train_batch_size=1   \
    --learning_rate 4e-5 \
    --warmup_steps 0.1 \
    --overwrite_output_dir \
    --logging_steps 50 \
    --num_workers 1 \
    --warmup_steps 0.05 \
    --max_length 1000 \
    --seed 10 \
    --output_dir $4 \
    --change_positional_embedding_after_loading \
     --num_train_epochs 20.0 \
    --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
    --train_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_train \--val_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_val




bash scripts/5_train_bert.bash \
 0,1,2,3 4 9820 \
 OUTPUT/LP/BERT/ \
 --num_train_epochs 20.0 \
 --gradient_accumulation_steps 8 --per_gpu_train_batch_size=2 \
 --train_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_train --val_file_path DATA/LP/prop_examples.balanced_by_backward.max_6.json_val
