def get_dataset_download_url(cfg):
    env = cfg.env
    # Gym
    if env == "hopper-medium-v2":
        return "https://drive.google.com/drive/u/1/folders/18Ti-92XVq3sE24K096WAxjC_SCCngeHd"
    elif env == "walker2d-medium-v2":
        return "https://drive.google.com/drive/u/1/folders/1BJu8NklriunDHsDrLT6fEpcro3_2IPFf"
    elif env == "halfcheetah-medium-v2":
        return "https://drive.google.com/drive/u/1/folders/1Drel26tiuQ9oD3YNf1eyy0UVaf5SQj-U"
    # D3IL
    elif env == "avoid" and cfg.mode == "d56_r12":  # M1
        return "https://drive.google.com/drive/u/1/folders/1ZAPvLQwv2y4Q98UDVKXFT4fvGF5yhD_o"
    elif env == "avoid" and cfg.mode == "d57_r12":  # M2
        return "https://drive.google.com/drive/u/1/folders/1wyJi1Zbnd6JNy4WGszHBH40A0bbl-vkd"
    elif env == "avoid" and cfg.mode == "d58_r12":  # M3
        return "https://drive.google.com/drive/u/1/folders/1mNXCIPnCO_FDBlEj95InA9eWJM2XcEEj"
    # Robomimic
    elif env == "lift" and "img" not in cfg.train_dataset_path:  # state
        return "https://drive.google.com/drive/u/1/folders/1lbXgMKBTAiFdJqPZqWXpwjEyrVW16MBu"
    elif env == "lift" and "img" in cfg.train_dataset_path:  # img
        return "https://drive.google.com/drive/u/1/folders/1H-UncdzHx6wd5NWVzrQyftfGls7KGz1O"
    elif env == "can" and "img" not in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1J1qSvsDEf40jnMZY9W0r6ww--E3MdmK3"
    elif env == "can" and "img" in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1VGp_5xXXb1-GJutdSc6AZSzXNk-6_vRz"
    elif env == "square" and "img" not in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1mVVNOJ6wt2EXoapF7PKkcqxbsB9gvK-B"
    elif env == "square" and "img" in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1-aGqVeKLIzJCEst8p0ZTjfjkrXFfJLxa"
    elif env == "transport" and "img" not in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1EVHmFx-YdX4MEE1EjwduVayvaH9vvqvK"
    elif env == "transport" and "img" in cfg.train_dataset_path:
        return "https://drive.google.com/drive/u/1/folders/1cOkAZQmmETYEPFrnnX0EuD6mv0kUfMO2"
    # Furniture-Bench
    elif env == "one_leg_low_dim":
        return "https://drive.google.com/drive/u/1/folders/1v4LG2D1fS8id5hqE7Jjt7MYNFUEBNyh4"
    elif env == "one_leg_med_dim":
        return "https://drive.google.com/drive/u/1/folders/1ohDuMSgCqGN1CSh1cI_8A0ia3DVHzj3w"
    elif env == "lamp_low_dim":
        return "https://drive.google.com/drive/u/1/folders/14MqDUmuNmTFcBtKw5gcvx7nuir0zmF7V"
    elif env == "lamp_med_dim":
        return "https://drive.google.com/drive/u/1/folders/1bhOoN0xet4ga0rOIvcRHUFoaCjPRNRLf"
    elif env == "round_table_low_dim":
        return "https://drive.google.com/drive/u/1/folders/15oF3qiqGzlT_98FDoTIVtGmLtHd5SIbi"
    elif env == "round_table_med_dim":
        return "https://drive.google.com/drive/u/1/folders/1U27xjdRrKlLC8E33o7jMFZ1HF5P_Soik"
    # unknown
    else:
        raise ValueError(f"Unknown environment {env}")


def get_normalization_download_url(cfg):
    env = cfg.env_name
    # Gym
    if env == "hopper-medium-v2":
        return "https://drive.google.com/file/d/1HHZ2X6r5io6hjG-MHVFFoPMJV2fJ3lis/view?usp=drive_link"
    elif env == "walker2d-medium-v2":
        return "https://drive.google.com/file/d/1NSX7t3DFKaBj5HNpv91Oo5h6oXTk0zoo/view?usp=drive_link"
    elif env == "halfcheetah-medium-v2":
        return "https://drive.google.com/file/d/1LlwCMfy1b5e8jSx99CV3lWhcrQWrI2Jm/view?usp=drive_link"
    # D3IL
    elif env == "avoiding-m5" and cfg.mode == "d56_r12":  # M1
        return "https://drive.google.com/file/d/1PubKaPabbiSdWYpGmouDhYfXp4QwNHFG/view?usp=drive_link"
    elif env == "avoiding-m5" and cfg.mode == "d57_r12":  # M2
        return "https://drive.google.com/file/d/1Hoohw8buhsLzXoqivMA6IzKS5Izlj07_/view?usp=drive_link"
    elif env == "avoiding-m5" and cfg.mode == "d58_r12":  # M3
        return "https://drive.google.com/file/d/1qt7apV52C9Tflsc-A55J6uDMHzaFa1wN/view?usp=drive_link"
    # Robomimic
    elif env == "lift" and "img" not in cfg.normalization_path:  # state
        return "https://drive.google.com/file/d/1d3WjwRds-7I5bBFpZuY27OT9ycb8r_QM/view?usp=drive_link"
    elif env == "lift" and "img" in cfg.normalization_path:  # img
        return "https://drive.google.com/file/d/15GnKDIK8VasvUHahcvEeK_uEs1J9i0ja/view?usp=drive_link"
    elif env == "can" and "img" not in cfg.normalization_path:
        return "https://drive.google.com/file/d/14FxHk9zQ-5ulAO26a6xrdvc-gRkL36FR/view?usp=drive_link"
    elif env == "can" and "img" in cfg.normalization_path:
        return "https://drive.google.com/file/d/1APAB6W10ECaNVVL72F0C2wC6oJhhbjWX/view?usp=drive_link"
    elif env == "square" and "img" not in cfg.normalization_path:
        return "https://drive.google.com/file/d/1FFMqWVv0145OJjbA_iglkWywmdK22Za-/view?usp=drive_link"
    elif env == "square" and "img" in cfg.normalization_path:
        return "https://drive.google.com/file/d/1jq5atfHdu-ZMQ8YaFjwcctbJESBYDgP8/view?usp=drive_link"
    elif env == "transport" and "img" not in cfg.normalization_path:
        return "https://drive.google.com/file/d/1EmC80gIgLoqQ8kRPH5r0mDVqX6NqHO3p/view?usp=drive_link"
    elif env == "transport" and "img" in cfg.normalization_path:
        return "https://drive.google.com/file/d/1LBgvIacNzbXCZXWKYanddqiotevG3hmA/view?usp=drive_link"
    # Furniture-Bench
    elif env == "one_leg_low_dim":
        return "https://drive.google.com/file/d/1fbYxau8Z0tifeuu_06UKdRRz8zozxMUE/view?usp=drive_link"
    elif env == "one_leg_med_dim":
        return "https://drive.google.com/file/d/1bT-MIG99-uwSXL3VwCz6kK9Kg9t6a7BZ/view?usp=drive_link"
    elif env == "lamp_low_dim":
        return "https://drive.google.com/file/d/1KRyCcCT63-cikz6Q_McXM0iVYx9M2E_o/view?usp=drive_link"
    elif env == "lamp_med_dim":
        return "https://drive.google.com/file/d/1WBHKbA6BtSfF3qfRIkyuWga5T9X3fEvg/view?usp=drive_link"
    elif env == "round_table_low_dim":
        return "https://drive.google.com/file/d/1CwfNLI5KXEkl_jVgtAY7-xfq7jED-xc0/view?usp=drive_link"
    elif env == "round_table_med_dim":
        return "https://drive.google.com/file/d/1vNxzok6f6HROGxQR2eoThsN62hVjpayn/view?usp=drive_link"
    # unknown
    else:
        raise ValueError(f"Unknown environment {env}")


def get_checkpoint_download_url(cfg):
    path = cfg.base_policy_path
    ######################################
    ####             Gym
    ######################################
    if (
        "hopper-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-10-05/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1uV3beg2YuBRh11t7jnRsyd_M9bsFGOkA/view?usp=drive_link"
    elif (
        "walker2d-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-06-12/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1StopetttozWba4l9u0VT_JW1rYe6kJx2/view?usp=drive_link"
    elif (
        "halfcheetah-medium-v2_pre_diffusion_mlp_ta4_td20/2024-06-12_23-04-42/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1o9ryyeZQAsaB4ffUTCJkIaGCi0frL3G4/view?usp=drive_link"
    ######################################
    ####             D3IL
    ######################################
    elif (
        "avoid_d56_r12_pre_diffusion_mlp_ta4_td20/2024-07-06_22-50-07/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1JdEOG0KsCA9EX9zq09DE0xkTB4xy6DNp/view?usp=drive_link"
    elif (
        "avoid_d56_r12_pre_gaussian_mlp_ta4/2024-07-07_01-35-48/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/138wEi_rVV5HpcwgH6_3BlXQ1dhgZN05L/view?usp=drive_link"
    elif (
        "avoid_d56_r12_pre_gmm_mlp_ta4/2024-07-10_14-30-00/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1krvEINP6UBfnJG9bEge53h3MNJkTQXe_/view?usp=drive_link"
    elif (
        "avoid_d57_r12_pre_diffusion_mlp_ta4_td20/2024-07-07_13-12-09/checkpoint/state_15000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1wAmRHzRZ4O5z_ZWDhVg5JFZbugo4cHKA/view?usp=drive_link"
    elif (
        "avoid_d57_r12_pre_gaussian_mlp_ta4/2024-07-07_02-15-50/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1hf_647bJ0EMRhArsfxStSkhgkGaWGhmD/view?usp=drive_link"
    elif (
        "avoid_d57_r12_pre_gmm_mlp_ta4/2024-07-10_15-44-32/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1CE4AcNJp2UITIHpuLUwxp7cDH8Y8jsYC/view?usp=drive_link"
    elif (
        "avoid_d58_r12_pre_diffusion_mlp_ta4_td20/2024-07-07_13-54-54/checkpoint/state_15000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1w5X1lJZd0wI6E2XRdj839TqJ_9EziGRx/view?usp=drive_link"
    elif (
        "avoid_d58_r12_pre_gaussian_mlp_ta4/2024-07-07_13-11-49/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1YIVEN0Ykica9dj_DxxPrB4QJLgv-Ne0N/view?usp=drive_link"
    elif (
        "avoid_d58_r12_pre_gmm_mlp_ta4/2024-07-10_17-01-50/checkpoint/state_10000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/174tadrqjfxJdOgsMjNbg53goZtIkpW3f/view?usp=drive_link"
    ######################################
    ####             Robomimic-Lift
    ######################################
    elif (
        "lift_pre_diffusion_unet_ta4_td20/2024-06-29_02-49-45/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1T-NGgBmT-UmcVWADygXj873IyWLewvsU/view?usp=drive_link"
    elif (
        "lift_pre_diffusion_mlp_ta4_td20/2024-06-28_14-47-58/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Ngr-DNxoB9XNCZ2O-NF5p60NzmYlzmWG/view?usp=drive_link"
    elif (
        "lift_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-24-35/checkpoint/state_2500.pt"
        in path
    ):
        return "https://drive.google.com/file/d/19hqNicwKKKDrlS5UMr-FLRu51EAW8Z51/view?usp=drive_link"
    elif (
        "lift_pre_gaussian_mlp_ta4/2024-06-28_14-48-24/checkpoint/state_5000.pt" in path
    ):
        return "https://drive.google.com/file/d/157x5_XJy3ZyaPz_7Vr_opXAZ_mPvlmhp/view?usp=drive_link"
    elif (
        "lift_pre_gaussian_mlp_img_ta4/2024-07-28_23-00-48/checkpoint/state_500.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Uae7K3Hv9XzaAGljG2fjsxYyOEgL03zv/view?usp=drive_link"
    elif (
        "lift_pre_gaussian_transformer_ta4/2024-06-28_14-49-23/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1Z_C8ureDDPXqpDUaRMVKBt3VgfZlIQdE/view?usp=drive_link"
    elif "lift_pre_gmm_mlp_ta4/2024-06-28_15-30-32/checkpoint/state_5000.pt" in path:
        return "https://drive.google.com/file/d/1wFvBoIOaCJVibqEIYSjJxAbxjGkH7RMk/view?usp=drive_link"
    elif (
        "lift_pre_gmm_transformer_ta4/2024-06-28_14-51-23/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1w_3WOXS51debWc1ShgaO2VQx49dj9ky2/view?usp=drive_link"
    ######################################
    ####             Robomimic-Can
    ######################################
    elif (
        "can_pre_diffusion_unet_ta4_td20/2024-06-29_02-49-45/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1s346KCe2aar_tXX7u8rzjRF3kpwVpH5c/view?usp=drive_link"
    elif (
        "can_pre_diffusion_mlp_ta4_td20/2024-06-28_13-29-54/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1L1ZLD1u1Y1YJmRLGzScXbQ02wGS-_cWo/view?usp=drive_link"
    elif (
        "can_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-23-55/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1siIKDGVHld3ZH8vDqgu9iE5H9KoaYq_Z/view?usp=drive_link"
    elif (
        "can_pre_gaussian_mlp_ta4/2024-06-28_13-31-00/checkpoint/state_5000.pt" in path
    ):
        return "https://drive.google.com/file/d/1bA-A0p0KnHwrVO3MqjuYdV3dZuuZICMy/view?usp=drive_link"
    elif (
        "can_pre_gaussian_mlp_img_ta4/2024-07-28_21-54-40/checkpoint/state_1000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/16vUUyVO9DvnDyPtSGiZx4iQP4JxuSAoS/view?usp=drive_link"
    elif (
        "can_pre_gaussian_transformer_ta4/2024-06-28_13-42-20/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1cGf7s8aS5grZsGRb5PYZiPogcPRr_lrf/view?usp=drive_link"
    elif "can_pre_gmm_mlp_ta4/2024-06-28_13-32-19/checkpoint/state_5000.pt" in path:
        return "https://drive.google.com/file/d/1KVx8-KICiIHstcjsvhPlQc_pZZ6RxXpd/view?usp=drive_link"
    elif (
        "can_pre_gmm_transformer_ta4/2024-06-28_13-43-21/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1xSgwGG40zdoO2DDSM79l0rMHeNmaifnq/view?usp=drive_link"
    ######################################
    ####             Robomimic-Square
    ######################################
    elif (
        "square_pre_diffusion_unet_ta4_td20/2024-06-29_02-48-45/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/11IEgQe0LFI23hn1Cwf6Z_YfJdDilVc0z/view?usp=drive_link"
    elif (
        "square_pre_diffusion_mlp_ta4_td20/2024-07-10_01-46-16/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1lP9mNe2AxMigfOywcaHOOR7FxQ-KR_Ee/view?usp=drive_link"
    elif (
        "square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1miud5SX41xjPoW8yqClRhaFx0Q7EWl3V/view?usp=drive_link"
    elif (
        "square_pre_gaussian_mlp_ta4/2024-06-28_15-02-32/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1rETiyXLz7YgYoHKwLEZa7dFabu4gfJR8/view?usp=drive_link"
    elif (
        "square_pre_gaussian_mlp_img_ta4/2024-07-30_18-44-32/checkpoint/state_4000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1myB6FOAmt6c6x3ScGKXRULgx826tZVS4/view?usp=drive_link"
    elif (
        "square_pre_gaussian_transformer_ta4/2024-06-28_15-02-39/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1JJQ7KbRWWBB09PLwNRriAUEAl9vjFeCW/view?usp=drive_link"
    elif "square_pre_gmm_mlp_ta4/2024-06-28_15-03-08/checkpoint/state_5000.pt" in path:
        return "https://drive.google.com/file/d/10ujnnOk2Pn-yjE7iW9-RYbApo_aKiNRq/view?usp=drive_link"
    elif (
        "square_pre_gmm_transformer_ta4/2024-06-28_15-03-15/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1LczXhgeNtQfqySsfGNbbviPrlLwyh-E3/view?usp=drive_link"
    ######################################
    ####             Robomimic-Transport
    ######################################
    elif (
        "transport_pre_diffusion_unet_ta16_td20/2024-07-04_02-20-53/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1MNGT8j9x1uudugGUcia-xwP_7f7xVY4K/view?usp=drive_link"
    elif (
        "transport_pre_diffusion_mlp_ta8_td20/2024-07-08_11-18-59/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1240FmcDPg_VyXReEtePjBN4MML-OT21C/view?usp=drive_link"
    elif (
        "transport_pre_diffusion_mlp_img_ta8_td100/2024-07-30_22-30-06/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1_UmjNv27_w49XC_EO1WexgreJmC0SEdF/view?usp=drive_link"
    elif (
        "transport_pre_gaussian_mlp_ta8/2024-07-10_01-50-52/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1NOHDNHu1sTabxBd4DZrSrTyX9l1I2GuO/view?usp=drive_link"
    elif (
        "transport_pre_gaussian_mlp_img_ta8/2024-07-30_21-39-34/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1XYOIwOEgOVoUdMmxWuBRAcA5BOS777Kc/view?usp=drive_link"
    elif (
        "transport_pre_gaussian_transformer_ta8/2024-06-28_15-18-16/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1hnt9lX5bg82iFFsAk5FY3TktPXsAhUFU/view?usp=drive_link"
    elif (
        "transport_pre_gmm_mlp_ta8/2024-07-10_01-51-21/checkpoint/state_5000.pt" in path
    ):
        return "https://drive.google.com/file/d/1da9yLIu5ahq-ZgIIsG7wqehA5VopFTnt/view?usp=drive_link"
    elif (
        "transport_pre_gmm_transformer_ta8/2024-06-28_15-18-43/checkpoint/state_5000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1c0S7WX-U1Kn6n-wWZEQSR1alLPW2eZmi/view?usp=drive_link"
    ######################################
    ####             Furniture-One_leg
    ######################################
    elif (
        "one_leg_low_dim_pre_diffusion_mlp_ta8_td100/2024-07-22_20-01-16/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1tP0i53EMwNyw_bS2lt9WtQDIAoTxn40g/view?usp=drive_link"
    elif (
        "one_leg_low_dim_pre_diffusion_unet_ta16_td100/2024-07-03_22-23-38/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/13nOr7EI79RqdRuoc-je0LoTX5Y-IaO9J/view?usp=drive_link"
    elif (
        "one_leg_low_dim_pre_gaussian_mlp_ta8/2024-06-26_23-43-02/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1DeHj-IEX3a2ZLXFlV0MYe1N_5qupod_w/view?usp=drive_link"
    elif (
        "one_leg_med_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-11/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1aEzObl4EkOBKSs0wI2MGmkslzIjZy2_L/view?usp=drive_link"
    elif (
        "one_leg_med_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-16-16/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1pSwp_IUDSQ15OszChCkrrTQYdvG3pHc1/view?usp=drive_link"
    elif (
        "one_leg_med_dim_pre_gaussian_mlp_ta8/2024-06-28_16-27-02/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1lC4PKRPn4tRWR9VW5fSfCu3MKnRvYqD6/view?usp=drive_link"
    ######################################
    ####             Furniture-Lamp
    ######################################
    elif (
        "lamp_low_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-20/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1md5m4vpe5MmStu-fOk8snF-52BYGZpSc/view?usp=drive_link"
    elif (
        "lamp_low_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-16-48/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/103lUuvyKPvp97hzBYnUDAWgor6LsUf21/view?usp=drive_link"
    elif (
        "lamp_low_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-51/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1tK6NuZf4_xtTlZksIR9oQxiFEADFRVA9/view?usp=drive_link"
    elif (
        "lamp_med_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-20/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1yVUXYCK_vFhQ7mawxnCGxuHS6RgWWDjj/view?usp=drive_link"
    elif (
        "lamp_med_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-17-21/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1_qd47U50on-T1Pqojlfy7S3Cl6bLm_D1/view?usp=drive_link"
    elif (
        "lamp_med_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-56/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1d16HmHgidtXForqoN5QJg_e152XCaJM5/view?usp=drive_link"
    ######################################
    ####             Furniture-Round_table
    ######################################
    elif (
        "round_table_low_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-26/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1iJyqJr84AtszGqPeSysN70A-SmP73mKu/view?usp=drive_link"
    elif (
        "round_table_low_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-19-48/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1F3RFgLcFemU-IDUrLCXMzmQXknC2ISgJ/view?usp=drive_link"
    elif (
        "round_table_low_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-51/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1eX2u0cvu_zveblrg2htFomAxSToJmHyL/view?usp=drive_link"
    elif (
        "round_table_med_dim_pre_diffusion_mlp_ta8_td100/2024-07-23_01-28-29/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1DDvHEqb7WqVgT3_W6gYCNDQrvGyreatP/view?usp=drive_link"
    elif (
        "round_table_med_dim_pre_diffusion_unet_ta16_td100/2024-07-04_02-20-21/checkpoint/state_8000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1hzungyvt3Uc-XbrCzVghzdr59zDf6FlD/view?usp=drive_link"
    elif (
        "round_table_med_dim_pre_gaussian_mlp_ta8/2024-06-28_16-26-38/checkpoint/state_3000.pt"
        in path
    ):
        return "https://drive.google.com/file/d/1LvELZ7A-whxKk1Oq7S8M9uBiSrWPAGQT/view?usp=drive_link"
    # unknown --- this means the user trained own policy but specifies the wrong path
    else:
        return None
