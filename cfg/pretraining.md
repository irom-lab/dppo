## Pre-training experiments

### Comparing diffusion-based RL algorithms (Sec. 5.1)
Gym configs are under `cfg/gym/pretrain/<env_name>/`, and the config name is `pre_diffusion_mlp`. Robomimic configs are under `cfg/robomimic/pretrain/<env_name>/`, and the name is also `pre_diffusion_mlp`.

**Note**: In both Gym and Robomimic experiments, for the experiments in the paper we used more than enough expert demonstrations for pre-training. You can specify `+train_dataset.max_n_episodes=<number_of_episodes>` to limit the number of episodes so the pre-training is faster.

### Comparing policy parameterizations (Sec. 5.2, 5.3)

Robomimic configs are under `cfg/robomimic/pretrain/<env_name>/`, and the naming follows `pre_<diffusion/gaussian/gmm>_<mlp/unet/transformer>_<img?>`. Furniture-Bench configs are under `cfg/furniture/pretrain/<env_name>/`, and the naming follows `pre_<diffusion/gaussian>_<mlp/unet>`.

### D3IL (Sec. 6)

D3IL configs are under `cfg/d3il/pretrain/avoid_<mode>/`, and the naming follows `pre_<diffusion/gaussian/gmm>_mlp`. In the paper we manually examine the pre-trained checkpoints and pick the ones that visually match the pre-training data the best. We also tune the Gaussian and GMM policy architecture extensively for best pre-training performance. The action chunk size can be specified with `horizon_steps` and the number of denoising steps can be specified with `denoising_steps`.
