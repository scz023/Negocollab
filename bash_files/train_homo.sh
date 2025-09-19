export CUDA_VISIBLE_DEVICES=1

model_dir="checkpoints/opv2v_single_pyramid_m1_new1"
yaml_file="opencood/hypes_yaml/opv2v/Homo/m1_pointpillar_pyramid.yaml"

mkdir "${model_dir}"

cp "${yaml_file}" "${model_dir}/config.yaml"
python opencood/tools/train.py \
-y $yaml_file \
--model_dir $model_dir
