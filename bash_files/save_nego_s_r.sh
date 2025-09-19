export CUDA_VISIBLE_DEVICES=1

model_dir="checkpoints/opv2v_nego_m1m2_pyramid"

python opencood/tools/find_best_ap50_and_save_bak.py \
--eval_epoch 5 \
--model_dir $model_dir