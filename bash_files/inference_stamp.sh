export CUDA_VISIBLE_DEVICES=0

model_dir='checkpoints/zz-opv2v_stamp_from_m0/opv2v_stamp_pyramid_m1m2m3m4'
# yaml_file='opencood/hypes_yaml/opv2v/STAMP/protocol_pointpillar_pyramid/m1m3_pyramid.yaml'

# cp "${yaml_file}" "${model_dir}/config.yaml"
python opencood/tools/inference_stamp.py \
--model_dir $model_dir