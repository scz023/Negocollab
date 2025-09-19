export CUDA_VISIBLE_DEVICES=1

model_dir='checkpoints/zz-opv2v_mpda_protocol_from_m0/z-opv2v_mpda_protocol_m1m2m3m4_pyramid'
# yaml_file='opencood/hypes_yaml/opv2v/MPDA_Protocol/m1m2m3m4_pyramid.yaml'

# cp "${yaml_file}" "${model_dir}/config.yaml"
python opencood/tools/inference_da.py \
--model_dir $model_dir