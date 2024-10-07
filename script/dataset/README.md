## Data processing scripts

These are some scripts used for processing the raw datasets from the benchmarks. We already pre-processed them and provide the final datasets.

Gym and robomimic data
```console
python script/dataset/get_d4rl_dataset.py --env_name=hopper-medium-v2 --save_dir=data/gym/hopper-medium-v2
python script/dataset/process_robomimic_dataset.py --load_path=../robomimic_raw_data/lift_low_dim_v141.hdf5 --save_dir=data/robomimic/lift --normalize
```

Raw robomimic data can be downloaded with a clone of the repository and then
```console
cd ~/robomimic/robomimic/scripts
python download_datasets.py --tasks all --dataset_types mh --hdf5_types low_dim # state-only policy
python download_datasets.py --tasks all --dataset_types mh --hdf5_types raw # pixel-based policy
# for pixel, replay the trajectories to extract image observations
python robomimic/scripts/dataset_states_to_obs.py --done_mode 2 --dataset datasets/can/mh/demo_v141.hdf5 --output_name image_v141.hdf5 --camera_names robot0_eye_in_hand --camera_height 96 --camera_width 96 --exclude-next-obs --n 100
```