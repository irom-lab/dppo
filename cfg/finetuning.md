## Fine-tuning experiments

### Comparing diffusion-based RL algorithms (Sec. 5.1)
Gym configs are under `cfg/gym/finetune/<env_name>/`, and the naming follows `ft_<alg_name>_diffusion_mlp`, e.g., `ft_awr_diffusion_mlp`. `alg_name` is one of `rwr`, `awr`, `dipo`, `idql`, `dql`, `qsm`, `ppo` (DPPO), `ppo_exact` (exact likelihood). They share the same pre-trained checkpoint in each env.

Robomimic configs are under `cfg/robomimic/finetune/<env_name>/`, and the naming follows the same.

<!-- **Note**: For *Can* and *Lift* in Robomimic, we use earlier checkpoints from pre-training (epoch 5000) so there is more room for fine-tuning improvement. For comparing policy parameterizations, we use the final checkpoints for all tasks (epoch 8000). -->

### Comparing policy parameterizations (Sec. 5.2, 5.3)

Robomimic configs are under `cfg/robomimic/finetune/<env_name>/`, and the naming follows `ft_ppo_<diffusion/gaussian/gmm>_<mlp/unet/transformer>_<img?>`. For pixel experiments, we choose pre-trained checkpoints such that the pre-training performance is similar between DPPO and Gaussian baseline.

**Note**: For *Can* and *Lift* in Robomimic with DPPO, you need to manually download the final checkpoints (epoch 8000). The default ones in the configs are from epoch 5000 (more room for fine-tuning improvement) and used for comparing diffusion-based RL algorithms, 

Furniture-Bench configs are under `cfg/furniture/finetune/<env_name>/`, and the naming follows `ft_<diffusion/gaussian>_<mlp/unet>`. In the paper we did not show the results of `ft_diffusion_mlp`. Running IsaacGym for the first time may take a while for setting up the meshes. If you encounter the error about `libpython`, see instruction [here](installation/install_furniture.md#L27).

### D3IL (Sec. 6)

D3IL configs are under `cfg/d3il/finetune/avoid_<mode>/`, and the naming follows `ft_ppo_<diffusion/gaussian/gmm>_mlp`. The number of fine-tuned denoising steps can be specified with `ft_denoising_steps`.

### Training from scratch (App. B.2)
`ppo_diffusion_mlp` and `ppo_gaussian_mlp` under `cfg/gym/finetune/<env_name>` are for training DPPO or Gaussian policy from scratch.

### Comparing to exact likelihood policy gradient (App. B.5)
`ft_ppo_exact_diffusion_mlp` under `cfg/gym/finetune/hopper-v2`, `cfg/gym/finetune/halfcheetah-v2`, and `cfg/robomimic/finetune/can` are for training diffusion policy gradient with exact likelihood. `torchdiffeq` package needs to be installed first with `pip install -e .[exact]`.
