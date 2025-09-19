export CUDA_VISIBLE_DEVICES=0

model_dir="checkpoints/zz-opv2v_nego_from_m1m2/z-opv2v_nego_m1m2_pyramid-ft"
# yaml_file='opencood/hypes_yaml/opv2v/NGCB/stage1/m1m2_pyramid.yaml'


cp "${yaml_file}" "${model_dir}/config.yaml"

python opencood/tools/train_nego_onlytrain.py \
-y None \
-s ft \
--model_dir $model_dir \
--val True \
--gpu_for_test 1