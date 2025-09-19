export CUDA_VISIBLE_DEVICES=1


model_dir="checkpoints/zz-opv2v_nego_from_m1/z-opv2v_nego_m1m2_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/inference/m1m2_pyramid.yaml'

model_dir="checkpoints/zz-opv2v_nego_from_m1m2/z-opv2v_nego_m1m3_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/inference/m1m3_pyramid.yaml'

model_dir="checkpoints/zz-opv2v_nego_from_m1m2/z-opv2v_nego_m2m4_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/inference/m2m4_pyramid.yaml'

model_dir="checkpoints/zz-opv2v_nego_from_m1m2/z-opv2v_nego_m3m4_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/inference/m3m4_pyramid.yaml'

model_dir="checkpoints/zz-opv2v_nego_from_m1m2/z-opv2v_nego_m1m2m3m4_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/inference/m1m2m3m4_pyramid.yaml'



export CUDA_VISIBLE_DEVICES=1

# cp "${yaml_file}" "${model_dir}/config.yaml"

python opencood/tools/inference_nego.py \
--model_dir $model_dir
