#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

for LEARNING_RATE in 2e-5 1e-5; do
  for EPOCHS in 2.0 3.0 4.0 5.0; do
        for TASK_NAME in "ours"; do
            for context_size in "gold" "1" "3" "5" "7" "9"; do
                for K in 0; do
                    DATA_DIR="./data/${TASK_NAME}_${context_size}_context/"
                    SEED=100
                    BATCH_SIZE=8
                    OUTPUT_DIR="./output/st-roberta-large_${TASK_NAME}_${context_size}_context_${EPOCHS}_${LEARNING_RATE}_${K}_${SEED}/"


                    ./../miniconda3/bin/python run_citation_classification.py \
                      --model_name_or_path roberta-large \
                      --model_type roberta \
                      --task_name ${TASK_NAME} \
                      --do_train \
                      --do_eval \
                      --data_dir ${DATA_DIR} \
                      --max_seq_length 512 \
                      --per_gpu_train_batch_size ${BATCH_SIZE} \
                      --learning_rate ${LEARNING_RATE} \
                      --num_train_epochs ${EPOCHS} \
                      --output_dir ${OUTPUT_DIR} \
                      --seed ${SEED} \
                      --classification_type multilabel \
                      --overwrite_cache \
                      --overwrite_output_dir \
                      --gradient_accumulation_steps 4 \
                      --save_steps -1 \
                      --k ${K}

                    ./../miniconda3/bin/python run_citation_classification.py \
                      --model_name_or_path ${OUTPUT_DIR} \
                      --model_type roberta \
                      --task_name ${TASK_NAME} \
                      --do_test \
                      --data_dir ${DATA_DIR} \
                      --max_seq_length 512 \
                      --per_gpu_train_batch_size ${BATCH_SIZE} \
                      --learning_rate ${LEARNING_RATE} \
                      --num_train_epochs ${EPOCHS} \
                      --output_dir ${OUTPUT_DIR} \
                      --seed ${SEED} \
                      --overwrite_cache \
                      --classification_type multilabel \
                      --save_steps -1
                done;
            done;
        done;
    done;
done;