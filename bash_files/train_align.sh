export CUDA_VISIBLE_DEVICES=1

model_dir="checkpoints/zz-opv2v_nego_from_m1m2/z-opv2v_nego_m1m2_guide_m3_pyramid"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/stage2/m1m2_guide_m3_pyramid.yaml'


cp "${yaml_file}" "${model_dir}/config.yaml"

python opencood/tools/train_nego_onlytrain.py \
-y None \
-s align \
--model_dir $model_dir \
--val True \
--gpu_for_test 0