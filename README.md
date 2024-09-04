# Diffusion Policy Policy Optimization (DPPO)

[[Paper](https://arxiv.org/abs/2409.00588)]&nbsp;&nbsp;[[Website](https://diffusion-ppo.github.io/)]

[Allen Z. Ren](https://allenzren.github.io/)<sup>1</sup>, [Justin Lidard](https://jlidard.github.io/)<sup>1</sup>, [Lars L. Ankile](https://ankile.com/)<sup>2,3</sup>, [Anthony Simeonov](https://anthonysimeonov.github.io/)<sup>3</sup><br>
[Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/)<sup>3</sup>, [Anirudha Majumdar](https://mae.princeton.edu/people/faculty/majumdar)<sup>1</sup>, [Benjamin Burchfiel](http://www.benburchfiel.com/)<sup>4</sup>, [Hongkai Dai](https://hongkai-dai.github.io/)<sup>4</sup>, [Max Simchowitz](https://msimchowitz.github.io/)<sup>3,5</sup>

<sup>1</sup>Princeton University, <sup>2</sup>Harvard University, <sup>3</sup>Masschusetts Institute of Technology<br>
<sup>4</sup>Toyota Research Institute, <sup>5</sup>Carnegie Mellon University

<img src="https://github.com/diffusion-ppo/diffusion-ppo.github.io/blob/main/img/overview-full.png" alt="drawing" width="100%"/>

> DPPO is an algorithmic framework and set of best practices for fine-tuning diffusion-based policies in continuous control and robot learning tasks.

<!-- ## Release

* [08/30/24] [DPPO](https://diffusion-ppo.github.io/) codebase and technical whitepaper are released.  -->

## Installation 

1. Clone the repository
```console
git clone git@github.com:irom-lab/dppo.git
cd dppo
```

2. Install core dependencies with a conda environment (if you do not plan to use Furniture-Bench, a higher Python version such as 3.10 can be installed instead) on a Linux machine with a Nvidia GPU.
```console
conda create -n dppo python=3.8 -y
conda activate dppo
pip install -e .
```

3. Install specific environment dependencies (Gym / Robomimic / D3IL / Furniture-Bench) or all dependencies
```console
pip install -e .[gym] # or [robomimic], [d3il], [furniture]
pip install -e .[all]
```
<!-- **Note**: Please do not set macros for robomimic and robosuite that the warnings suggest --- we will use some different global variables than the ones defined in macro.py  -->

4. [Install MuJoCo for Gym and/or Robomimic](installation/install_mujoco.md). [Install D3IL](installation/install_d3il.md). [Install IsaacGym and Furniture-Bench](installation/install_furniture.md)

5. Set environment variables for data and logging directory (default is `data/` and `log/`), and set WandB entity (username or team name)
```
source script/set_path.sh
```

## Usage - Pre-training

**Note**: You may skip pre-training if you would like to use the default checkpoint (available for download) for fine-tuning.

<!-- ### Prepare pre-training data

First create a directory as the parent directory of the pre-training data and set the environment variable for it.
```console
export DPPO_DATA_DIR=/path/to/data -->
<!-- ``` -->

Pre-training data for all tasks are pre-processed and can be found at [here](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link). Pre-training script will download the data (including normalization statistics) automatically to the data directory.
<!-- The data path follows `${DPPO_DATA_DIR}/<benchmark>/<task>/train.npz`, e.g., `${DPPO_DATA_DIR}/gym/hopper-medium-v2/train.pkl`. -->

### Run pre-training with data
All the configs can be found under `cfg/<env>/pretrain/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
<!-- To run pre-training, first set your WandB entity (username or team name) and the parent directory for logging as environment variables. -->
<!-- ```console
export DPPO_WANDB_ENTITY=<your_wandb_entity>
export DPPO_LOG_DIR=<your_prefered_logging_directory>
``` -->
```console
# Gym - hopper/walker2d/halfcheetah
python script/train.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/gym/pretrain/hopper-medium-v2
# Robomimic - lift/can/square/transport
python script/train.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/can
# D3IL - avoid_m1/m2/m3
python script/train.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/d3il/pretrain/avoid_m1
# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/train.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/furniture/pretrain/one_leg_low
```

See [here](cfg/pretraining.md) for details of the experiments in the paper.


## Usage - Fine-tuning

<!-- ### Set up pre-trained policy -->

<!-- If you did not set the environment variables for pre-training, we need to set them here for fine-tuning. 
```console
export DPPO_WANDB_ENTITY=<your_wandb_entity>
export DPPO_LOG_DIR=<your_prefered_logging_directory>
``` -->
<!-- First create a directory as the parent directory of the downloaded checkpoints and set the environment variable for it.
```console
export DPPO_LOG_DIR=/path/to/checkpoint
``` -->

Pre-trained policies used in the paper can be found [here](https://drive.google.com/drive/folders/1ZlFqmhxC4S8Xh1pzZ-fXYzS5-P8sfpiP?usp=drive_link). Fine-tuning script will download the default checkpoint automatically to the logging directory.
 <!-- or you may manually download other ones (different epochs) or use your own pre-trained policy if you like. -->

 <!-- e.g., `${DPPO_LOG_DIR}/gym-pretrain/hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-08-26_22-31-03_42/checkpoint/state_0.pt`. -->

<!-- The checkpoint path follows `${DPPO_LOG_DIR}/<benchmark>/<task>/.../<run>/checkpoint/state_<epoch>.pt`. -->

### Fine-tuning pre-trained policy

All the configs can be found under `cfg/<env>/finetune/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
<!-- Running them will download the default pre-trained policy. -->
<!-- Running the script will download the default pre-trained policy checkpoint specified in the config (`base_policy_path`) automatically, as well as the normalization statistics, to `DPPO_LOG_DIR`.  -->
```console
# Gym - hopper/walker2d/halfcheetah
python script/train.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/gym/finetune/hopper-v2
# Robomimic - lift/can/square/transport
python script/train.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/robomimic/finetune/can
# D3IL - avoid_m1/m2/m3
python script/train.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/d3il/finetune/avoid_m1
# Furniture-Bench - one_leg/lamp/round_table_low/med
python script/train.py --config-name=ft_ppo_diffusion_mlp \
    --config-dir=cfg/furniture/finetune/one_leg_low
```

**Note**: In Gym, Robomimic, and D3IL tasks, we run 40, 50, and 50 parallelized MuJoCo environments on CPU, respectively. If you would like to use fewer environments (given limited CPU threads, or GPU memory for rendering), you can reduce `env.n_envs` and increase `train.n_steps`, so the total number of steps collected in each iteration (n_envs x n_steps) remains roughly the same. Try to set `train.n_steps` a multiple of `env.max_episode_steps`, and be aware that we only count episodes finished within an iteration for eval. Furniture-Bench tasks run IsaacGym on a single GPU.

To fine-tune your own pre-trained policy instead, override `base_policy_path` to your own checkpoint, which is saved under `checkpoint/` of the pre-training directory. You can set `base_policy_path=<path>` in the command line when launching fine-tuning.

<!-- **Note**: If you did not download the pre-training [data](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link), you need to download the normalization statistics from it for fine-tuning, e.g., `${DPPO_DATA_DIR}/furniture/round_table_low/normalization.pkl`. -->

See [here](cfg/finetuning.md) for details of the experiments in the paper.


### Visualization
* Furniture-Bench tasks can be visualized in GUI by specifying `env.specific.headless=False` and `env.n_envs=1` in fine-tuning configs.
* D3IL environment can be visualized in GUI by `+env.render=True`, `env.n_envs=1`, and `train.render.num=1`. There is a basic script at `script/test_d3il_render.py`.
* Videos of trials in Robomimic tasks can be recorded by specifying `env.save_video=True`, `train.render.freq=<iterations>`, and `train.render.num=<num_video>` in fine-tuning configs.

## DPPO implementation

Our diffusion implementation is mostly based on [Diffuser](https://github.com/jannerm/diffuser) and at [`model/diffusion/diffusion.py`](model/diffusion/diffusion.py) and [`model/diffusion/diffusion_vpg.py`](model/diffusion/diffusion_vpg.py). PPO specifics are implemented at [`model/diffusion/diffusion_ppo.py`](model/diffusion/diffusion_ppo.py). The main training script is at [`agent/finetune/train_ppo_diffusion_agent.py`](agent/finetune/train_ppo_diffusion_agent.py) that follows [CleanRL](https://github.com/vwxyzjn/cleanrl).

### Key configurations
* `denoising_steps`: number of denoising steps (should always be the same for pre-training and fine-tuning regardless the fine-tuning scheme)
* `ft_denoising_steps`: number of fine-tuned denoising steps
* `horizon_steps`: predicted action chunk size (should be the same as `act_steps`, executed action chunk size, with MLP. Can be different with UNet, e.g., `horizon_steps=16` and `act_steps=8`)
* `model.gamma_denoising`: denoising discount factor
* `model.min_sampling_denoising_std`: <img src="https://latex.codecogs.com/gif.latex?\epsilon^\text{exp}_\text{min} "/>, minimum amount of noise when sampling at a denoising step
* `model.min_logprob_denoising_std`: <img src="https://latex.codecogs.com/gif.latex?\epsilon^\text{prob}_\text{min} "/>, minimum standard deviation when evaluating likelihood at a denoising step
* `model.clip_ploss_coef`: PPO clipping ratio

### DDIM fine-tuning

To use DDIM fine-tuning, set `denoising_steps=100` in pre-training and set `model.use_ddim=True`, `model.ddim_steps` to the desired number of total DDIM steps, and `ft_denoising_steps` to the desired number of fine-tuned DDIM steps. In our Furniture-Bench experiments we use `denoising_steps=100`, `model.ddim_steps=5`, and `ft_denoising_steps=5`.

## Adding your own dataset/environment

### Pre-training data
Pre-training script is at [`agent/pretrain/train_diffusion_agent.py`](agent/pretrain/train_diffusion_agent.py). The pre-training dataset [loader](agent/dataset/sequence.py) assumes a pickle file containing a dictionary of `observations`, `actions`, and `traj_length`, where `observations` and `actions` have the shape of num_episode x max_episode_length x obs_dim/act_dim, and `traj_length` is a 1-D array. One pre-processing example can be found at [`script/process_robomimic_dataset.py`](script/process_robomimic_dataset.py).

**Note:** The current implementation does not support loading history observations (only using observation at the current timestep). If needed, you can modify [here](agent/dataset/sequence.py#L130-L131).

### Fine-tuning environment
We follow the Gym format for interacting with the environments. The vectorized environments are initialized at [make_async](env/gym_utils/__init__.py#L10) (called in the parent fine-tuning agent class [here](agent/finetune/train_agent.py#L38-L39)). The current implementation is not the cleanest as we tried to make it compatible with Gym, Robomimic, Furniture-Bench, and D3IL environments, but it should be easy to modify and allow using other environments. We use [multi_step](env/gym_utils/wrapper/multi_step.py) wrapper for history observations (not used currently) and multi-environment-step action execution. We also use environment-specific wrappers such as [robomimic_lowdim](env/gym_utils/wrapper/robomimic_lowdim.py) and [furniture](env/gym_utils/wrapper/furniture.py) for observation/action normalization, etc. You can implement a new environment wrapper if needed.

## Known issues
* IsaacGym simulation can become unstable at times and lead to NaN observations in Furniture-Bench. The current env wrapper does not handle NaN observations.

## License
This repository is released under the MIT license. See [LICENSE](LICENSE).

## Acknowledgement
* [Diffuser, Janner et al.](https://github.com/jannerm/diffuser): general code base and DDPM implementation
* [Diffusion Policy, Chi et al.](https://github.com/real-stanford/diffusion_policy): general code base especially the env wrappers
* [CleanRL, Huang et al.](https://github.com/vwxyzjn/cleanrl): PPO implementation
* [IBRL, Hu et al.](https://github.com/hengyuan-hu/ibrl): ViT implementation
* [D3IL, Jia et al.](https://github.com/ALRhub/d3il): D3IL benchmark
* [Robomimic, Mandlekar et al.](https://github.com/ARISE-Initiative/robomimic): Robomimic benchmark
* [Furniture-Bench, Heo et al.](https://github.com/clvrai/furniture-bench): Furniture-Bench benchmark
* [AWR, Peng et al.](https://github.com/xbpeng/awr): DAWR baseline (modified from AWR)
* [DIPO, Yang et al.](https://github.com/BellmanTimeHut/DIPO): DIPO baseline
* [IDQL, Hansen-Estruch et al.](https://github.com/philippe-eecs/IDQL): IDQL baseline
* [DQL, Wang et al.](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL): DQL baseline
* [QSM, Psenka et al.](https://www.michaelpsenka.io/qsm/): QSM baseline
* [Score SDE, Song et al.](https://github.com/yang-song/score_sde_pytorch/): diffusion exact likelihood