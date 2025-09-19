# NegoCollab
NegoCollab: A Common Representation Negotiation Approach for Heterogeneous Collaborative Perception [NeurIPS 2025]

## Repo Feature

- Modality Support
  - [x] LiDAR
  - [x] Camera
  - [x] LiDAR + Camera

- Heterogeneity Support
  - [x] **Sensor Data Heterogeneity**: We have multiple LiDAR data (16/32/64-line) and camera data (w./w.o. depth sensor) in the same scene.
  - [x] **Modality Heterogeneity**: You can assign different sensor modality to agents in the way you like!
  - [x] **Model Heterogeneity**: You can assign different model encoders (together with modality) to agents in the way you like!

- Dataset Support
  - [x] OPV2V
  - [x] V2XSet
  - [x] V2X-Sim 2.0
  - [x] DAIR-V2X-C

- Detector Support
  - [x] PointPillars (LiDAR)
  - [x] SECOND (LiDAR)
  - [x] Pixor (LiDAR)
  - [x] VoxelNet (LiDAR)
  - [x] Lift-Splat-Shoot (Camera)

- multiple heterogeneous collaborative perception methods
  - [x] [STAMP [ICLR2025]](https://arxiv.org/abs/2501.18616)
  - [x] [MPDA [ICDCS2022]](https://arxiv.org/abs/2210.08451)
  - [x] [PnPDA [ECCV2024]](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10564_ECCV_2024_paper.php)

## Data Preparation
- OPV2V: Please refer to [this repo](https://github.com/DerrickXuNu/OpenCOOD).
- OPV2V-H: We store our data in [Huggingface Hub](https://huggingface.co/datasets/yifanlu/OPV2V-H). Please refer to [Downloading datasets](https://huggingface.co/docs/hub/datasets-downloading) tutorial for the usage.
- V2XSet: Please refer to [this repo](https://github.com/DerrickXuNu/v2x-vit).
- V2X-Sim 2.0: Download the data from [this page](https://ai4ce.github.io/V2X-Sim/). Also download pickle files from [google drive](https://drive.google.com/drive/folders/16_KkyjV9gVFxvj2YDCzQm1s9bVTwI0Fw?usp=sharing).
- DAIR-V2X-C: Download the data from [this page](https://thudair.baai.ac.cn/index). We use complemented annotation, so please also follow the instruction of [this page](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/). 

Note that you can select your interested dataset to download. **OPV2V** and **DAIR-V2X-C** are heavily used in this repo, so it is recommended that you download and try them first. 

Create a `dataset` folder under `NegoCollab` and put your data there. Make the naming and structure consistent with the following:
```
NegoCollab/dataset

. 
├── my_dair_v2x 
│   ├── v2x_c
│   ├── v2x_i
│   └── v2x_v
├── OPV2V
│   ├── additional
│   ├── test
│   ├── train
│   └── validate
├── OPV2V_Hetero
│   ├── test
│   ├── train
│   └── validate
├── V2XSET
│   ├── test
│   ├── train
│   └── validate
├── v2xsim2-complete
│   ├── lidarseg
│   ├── maps
│   ├── sweeps
│   └── v1.0-mini
└── v2xsim2_info
    ├── v2xsim_infos_test.pkl
    ├── v2xsim_infos_train.pkl
    └── v2xsim_infos_val.pkl
```


## Installation
### Step 1: Basic Installation
```bash
conda create -n ngcb python=3.8
conda activate ngcb
# install pytorch. Cudatoolkit 11.3 are tested in our experiment.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# install dependency
pip install -r requirements.txt
# install this project. It's OK if EasyInstallDeprecationWarning shows up.
python setup.py develop
```


### Step 2: Install Spconv (1.2.1 or ~~2.x~~)
We use spconv 1.2.1 to generate voxel features. 

To install spconv 1.2.1, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.

#### Tips for installing spconv 1.2.1:
1. make sure your cmake version >= 3.13.2
2. CUDNN and CUDA runtime library (use `nvcc --version` to check) needs to be installed on your machine.

### Step 3: Bbx IoU cuda version compile
Install bbx nms calculation cuda version
  
```bash
python opencood/utils/setup.py build_ext --inplace
```

### Step 4: Install pypcd by hand for DAIR-V2X LiDAR loader.

``` bash
pip install git+https://github.com/klintan/pypcd.git
```

### Step 5: Dependencies for FPV-RCNN (optional)
Install the dependencies for fpv-rcnn.
  
```bash
cd NegoCollab
python opencood/pcdet_utils/setup.py build_ext --inplace
```


---
To align with our agent-type assignment in our experiments, please make a copy of the assignment file under the logs folder
```bash
# In NegoCollab directory
mkdir opencood/logs
cp -r opencood/modality_assign opencood/logs/heter_modality_assign
```

## New Style Yaml and Old Style Yaml

We introduced identifiers such as `m1`, `m2`, ... to indicate the modalities and models that an agent will use.  

However, yaml files without identifiers like `m1` (if you are familiar with the [CoAlign](https://github.com/yifanlu0227/CoAlign) repository) still work in this repository. For example, [PointPillar Early Fusion](https://github.com/yifanlu0227/CoAlign/blob/main/opencood/hypes_yaml/opv2v/lidar_only_with_noise/pointpillar_early.yaml). 

Note that there will be some differences in the weight key names of their two models' checkpoint. For example, training with the `m1` identifier will assign some parameters's name with prefix like `encoder_m1.`, `backbone_m1`, etc. But since the model structures are the same, you can convert them using the `rename_model_dict_keys` function in `opencood/utils/model_utils.py`.

### Agent type identifier

- The identifiers like `m1, m2` in `opv2v_4modality.json`  are used to assign agent type to each agent in the scene. With this assignment, we ensure the validation scenarios for all methods are consistent and fixed. To generate these json files, you can refer to [heter_utils.py](https://github.com/yifanlu0227/HEAL/blob/2fd71de77dada46ded8345aeb68026ce2346c214/opencood/utils/heter_utils.py#L96).

- The identifiers like `m1, m2` in `${METHOD}.yaml` are used to specify the sensor configuration and detection model used by this agent type (like `m2` in the case of `camera_pyramid.yaml`). 

In `${METHOD}.yaml`, there is also a concept of `mapping_dict`. It maps the given agent type of `opv2v_4modality.json` to the agent type in the current experiment. As you can see, `camera_pyramid.yaml` is a homogeneous collaborative perception setting, so the type of all agents should be the same, which can be referred to by `m2`.

Just note that `mapping_dict` will not take effect during the training process to introduce more data augmentation. Each agent will be randomly assigned an agent type that exists in the yaml.



<!-- 我们将NegoCollab、STAMP和MPDA-P、PnPDA-P训练完成的模型文件及推理所需文件保存在了目录`checkpoints`下,  -->



## Quick Start
The trained model files for NegoCollab, STAMP, MPDA-P, and PnPDA-P has uploaded to the [Hugging Face repository](https://huggingface.co/scz1/NegoCollab/tree/main). Create a `checkpoints` directory under the `NegoCollab/` folder, extract the downloaded model files `checkpoints.zip` into this `checkpoints` directory, then you can refer to the bash code in the corresponding script files for testing.

NegoCollab `inference_nego.sh` `checkpoints/zz-opv2v_nego_from_m1m2`\
STAMP `inference_stamp.sh` `checkpoints/zz-opv2v_stamp_from_m0` \
MPDA-P `inference_protocol.sh` `checkpoints/zz-opv2v_mpda_protocol_from_m0` \
PnPDA-P `inference_protocol.sh` `checkpoints/zz-opv2v_pnpda_protocol_from_m0`

## Training Process

### Step 1: Homogeneous Collaborative Training.

Suppose you are now in the `NegoCollab/` folder. Then train the homogeneous collaboration models for m1, m2, m3, and m4 separately. 
The following bash script demonstrates the training process for the m1 homogeneous model, while the training methods for m2, m3, and m4 are similar.
<!-- 随后分别训练m1、m2、m3、m4的同构协作模型. 以下bash脚本以m1的同构模型的训练作为示范, m2、m3、m4训练方法与其类似。 -->
```bash
# Create Directory
model_dir="checkpoints/opv2v_single_pyramid_m1"
yaml_file="opencood/hypes_yaml/opv2v/Homo/m1_pointpillar_pyramid.yaml"

mkdir "${model_dir}"
cp "${yaml_file}" "${model_dir}/config.yaml"

# Start Training
python opencood/tools/train.py \
-y $yaml_file \
--model_dir $model_dir
```

### Step 2: Initial Alliance Negotiation.
#### stage 1: Common Representation Negotiate
After completing the training of homogeneous collaboration model, we negotiate the common representation from the initial collaboration alliance. Taking the initial alliance formed by m1 and m2 as an example, first create the negotiation training folder `checkpoints/opv2v_nego_pyramid_m1m2`, then copy the trained homogeneous collaboration model files of m1 and m2 from the first step into this folder, rename them as `stage0_mx_collab`, and run `train_nego_onlytrain.py` to start training.

<!-- 同构协作模型训练完成后, 从初始协作联盟协商公共语义。
以从m1m2构成的初始联盟协商为例，首先创建协商训练文件夹`checkpoints/opv2v_nego_pyramid_m1m2`，随后将第一步训练完成的m1和m2同构协作模型文件复制到该文件夹下，重命名为 `stage0_mx_collab`, 运行`train_nego_onlytrain.py`开始训练。 -->

```bash
# Create Directory
model_dir="checkpoints/opv2v_nego_pyramid_m1m2"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/stage1/m1m2_pyramid.yaml'

mkdir "${model_dir}"
cp "${yaml_file}" "${model_dir}/config.yaml"

# Copy homogeneous collaborative models
cp checkpoints/opv2v_single_pyramid_m1 checkpoints/opv2v_nego_pyramid_m1m2/stage0_m1_collab
cp checkpoints/opv2v_single_pyramid_m2 checkpoints/opv2v_nego_pyramid_m1m2/stage0_m2_collab

# Start Training
python opencood/tools/train_nego_onlytrain.py \
-y None \
-s nego \
--model_dir $model_dir \
--val True \
--gpu_for_test 1
```

#### stage 2: Task Adaption
After completing the common representation negotiation, we adjust the receivers' parameters using the collaboration task loss. First, create an adaptation training folder `checkpoints/opv2v_nego_pyramid_m1m2-ft`, then copy the `stage0_mx_collab` and the best-performing AP model file from the stage1 training folder into this adaptation folder for continued training, enabling the model to adapt to downstream tasks.

<!-- 公共表征协商完成后, 使用协作任务损失对接收器参数进行调整。首先创建适配训练文件夹`checkpoints/opv2v_nego_pyramid_m1m2-ft`, 并将stage1训练文件夹中 `stage0_mx_collab`和AP表现最好的模型文件复制到适配训练文件夹下继续训练，使模型适应下游任务 -->

```bash
# Create Directory
model_dir="checkpoints/opv2v_nego_m1m2_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/stage1/m1m2_pyramid.yaml'

mkdir "${model_dir}"
cp "${yaml_file}" "${model_dir}/config.yaml"

# Copy the trained models of stage 1
cp checkpoints/opv2v_nego_pyramid_m1m2/stage0_m1_collab checkpoints/opv2v_nego_pyramid_m1m2-ft/stage0_m1_collab
cp checkpoints/opv2v_nego_pyramid_m1m2/stage0_m2_collab checkpoints/opv2v_nego_pyramid_m1m2-ft/stage0_m2_collab
cp checkpoints/opv2v_nego_pyramid_m1m2/net_epoch15.pth checkpoints/opv2v_nego_pyramid_m1m2-ft

# Start Training
python opencood/tools/train_nego_onlytrain.py \
-y None \
-s ft \
--model_dir $model_dir \
--val True \
--gpu_for_test 1
```

### Step 3: New Agent Alignment.
#### stage 1: Align to the Common Representaion
After completing the agent training within the initial alliance, we proceed to train new agents. The training utilizes the negotiator obtained from stage1 in step2, along with the sender and receiver from the initial alliance agents, to facilitate the alignment of the new agent's local representations to the common representations through the sender.

Taking the training for new agent m3 as an example, first use `find_best_ap50_and_save_bak.py` to extract and save the negotiator network from the best-performing epoch during negotiation into the `negotiator` directory, while saving the trained sender and receiver parameters of m1 and m2 into `stage0_mx_collab`. Then create a new agent alignment training folder, copy the `negotiator`, `stage0_mx_collab`, and rename the homogeneous collaboration model file of m3 as `stage0_m3_collab` into the alignment training directory.

<!-- 初始联盟内代理训练完成后, 即开始新代理的训练。使用step2中stage1训练得到协商器和初始联盟代理中的发送器和接收器辅助训练，使发送器可以将新代理的本地表征对齐到公共表征。
以新代理m3加入时的训练为例，首先使用`find_best_ap50_and_save_bak.py`提取协商过程中表现最佳epoch对应网络的协商器单独保存到`negotiator`目录下，并将训练好的m1和m2发送器和接收器的参数保存到 `stage0_mx_collab`中。随后创建新代理对齐训练文件夹，将协商器、`stage0_mx_collab`以及m3的同构协作模型文件更名为`stage0_m3_collab`复制到对齐训练目录下 -->

```bash

# Save Negotiator, senders and receivers
python opencood/tools/find_best_ap50_and_save_bak.py \
--eval_epoch 5(bestap) \
--model_dir checkpoints/opv2v_nego_m1m2_pyramid

# Create Directory
model_dir="checkpoints/opv2v_nego_m1m2_guide_m3_pyramid"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/stage2/m1m2_guide_m3_pyramid.yaml'

mkdir "${model_dir}"
cp "${yaml_file}" "${model_dir}/config.yaml"

# Copy the negotiator, sender and receiver from models trained in stage2.
cp checkpoints/opv2v_nego_m1m2/stage0_m1_collab checkpoints/opv2v_nego_m1m2_guide_m3_pyramid
cp checkpoints/opv2v_nego_m1m2/stage0_m2_collab checkpoints/opv2v_nego_m1m2_guide_m3_pyramid
cp checkpoints/opv2v_nego_m1m2/negotiator checkpoints/opv2v_nego_m1m2_guide_m3_pyramid

# Copy homogeneous collaborative models of m3
cp checkpoints/opv2v_single_pyramid_m3 checkpoints/opv2v_nego_m1m2_guide_m3_pyramid

python opencood/tools/train_nego_onlytrain.py \
-y None \
-s align \
--model_dir $model_dir \
--val True \
--gpu_for_test 0
```
#### stage 2: Task Adaption
After the new agent aligns with the common representations, the receiver parameters are fine-tuned using the collaboration task loss. 

First, create an adaptation training folder `checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft`. Then copy the best-performing model (by AP value) obtained from stage1 training into this adaptation folder to commence training.

<!-- 新代理对齐到公共表征后，通过协作任务损失对接收器参数进行调整。
首先创建适配训练文件夹`checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft`，随后将stage1中训练得到的AP值表现最好的模型复制到适配文件夹下，开始训练。 -->

```bash
# Create Directory
model_dir="checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/stage2/m1m2_guide_m3_pyramid.yaml'

mkdir "${model_dir}"
cp "${yaml_file}" "${model_dir}/config.yaml"

# Copy models trained in stage1
cp checkpoints/opv2v_nego_m1m2_guide_m3_pyramid/stage0_m1_collab checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft/stage0_m1_collab
cp checkpoints/opv2v_nego_m1m2_guide_m3_pyramid/stage0_m2_collab checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft/stage0_m2_collab
cp checkpoints/opv2v_nego_m1m2_guide_m3_pyramid/stage0_m3_collab checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft/stage0_m3_collab

cp checkpoints/opv2v_nego_m1m2_guide_m3_pyramid/net_epoch15.pth checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft

# Start Training
python opencood/tools/train_nego_onlytrain.py \
-y None \
-s ft \
--model_dir $model_dir \
--val True \
--gpu_for_test 0
```

## Inference
During inference, the agents are equipped with the sender and receiver after the training phase of task adaptation, and the heterogeneous collaborative performance in each scenario is tested.

After completing downstream task adaptation training for both the initial alliance and new agents, first execute `find_best_ap50_and_save_bak.py` to save the parameters of senders and receivers to the `stage0_mx_collab` directory. Then create an inference directory, copy both the scenario-specific yaml configuration files and all participating agents' `stage0_mx_collab` files into it, and run `inference_nego.py` to conduct the inference test.

Taking m1m2m3m4 collaborative inference as an example, the testing procedure is as follows:

<!-- 推理时，为智能体搭载任务适配训练阶段结束后的发送器和接收器，测试各场景下的异构协作表现。
初始联盟和新代理的下游任务适配阶段训练完成后，首先使用`find_best_ap50_and_save_bak.py`将发送器和接收器的参数保存到`stage0_mx_collab`目录下。随后，创建推理目录，并将场景对应的yaml配置文件和所有参与协作智能体的`stage0_mx_collab`复制到推理目录下，再运行`inference_nego.py`即可进行推理测试。以m1m2m3m4协作时的推理测试为例，推理测试过程如下-->

```bash
# Save the senders and receivers after the task adaptation training phase.
python opencood/tools/find_best_ap50_and_save_bak.py \
--eval_epoch 5(bestap) \
--model_dir checkpoints/opv2v_nego_m1m2_pyramid-ft

python opencood/tools/find_best_ap50_and_save_bak.py \
--eval_epoch 5(bestap) \
--model_dir checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft

python opencood/tools/find_best_ap50_and_save_bak.py \
--eval_epoch 5(bestap) \
--model_dir checkpoints/opv2v_nego_m1m2_guide_m4_pyramid-ft

# Create Directory
model_dir="checkpoints/opv2v_nego_m1m2m3m4_pyramid-ft"
yaml_file='opencood/hypes_yaml/opv2v/NGCB/inference/m1m2m3m4_pyramid.yaml'

mkdir "${model_dir}"
cp "${yaml_file}" "${model_dir}/config.yaml"

# Copy the agent models with senders and receivers to the inference directory
cp checkpoints/opv2v_nego_m1m2_pyramid-ft/stage0_m1_collab checkpoints/opv2v_nego_m1m2m3m4_pyramid-ft/stage0_m1_collab
cp checkpoints/opv2v_nego_m1m2_pyramid-ft/stage0_m2_collab checkpoints/opv2v_nego_m1m2m3m4_pyramid-ft/stage0_m2_collab
cp checkpoints/opv2v_nego_m1m2_guide_m3_pyramid-ft/stage0_m3_collab checkpoints/opv2v_nego_m1m2m3m4_pyramid-ft/stage0_m2_collab
cp checkpoints/opv2v_nego_m1m2_guide_m4_pyramid-ft/stage0_m4_collab checkpoints/opv2v_nego_m1m2m3m4_pyramid-ft/stage0_m4_collab

# Start Inference
python opencood/tools/inference_nego.py \
--model_dir $model_dir
```
