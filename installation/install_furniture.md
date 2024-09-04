## Install IsaacGym and Furniture-Bench (borrowed from [robust-rearrangement](https://github.com/ankile/robust-rearrangement/tree/main?tab=readme-ov-file#install-isaacgym)) 

### IsaacGym
Download the IsaacGym installer from the [IsaacGym website](https://developer.nvidia.com/isaac-gym) and follow the instructions to download the package by running:
* Click "Join now" and log into your NVIDIA account.
* Click "Member area".
* Read and check the box for the license agreement.
* Download and unzip Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release.

You can also download a copy of the file from their AWS S3 bucket for your convenience:
```console
cd ~
wget https://iai-robust-rearrangement.s3.us-east-2.amazonaws.com/IsaacGym_Preview_4_Package.tar.gz
```

Once the zipped file is downloaded, move it to the desired location and unzip it by running:
```console
tar -xzf IsaacGym_Preview_4_Package.tar.gz
```

Now, you can install the IsaacGym package by navigating to the isaacgym directory and running:
```
cd isaacgym
pip install -e python --no-cache-dir --force-reinstall
```
<!-- The --no-cache-dir and --force-reinstall flags are used to avoid potential issues with the installation we encountered. Also ignore Pip's notice that [notice] To update, run: pip install --upgrade pip as the current version of Pip is necessary for compatibility with the codebase. -->

**Note:** You will most likely encounter the error ```ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory``` when you run fine-tuning. You may fix it by (add to .bashrc if you want it permanently):
```console 
conda env list # print your conda_path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<conda_path>/envs/dppo/lib
```

### Furniture-Bench

Install our fork at your desired location:
```console
git clone git@github.com:ankile/furniture-bench.git
cd furniture-bench
pip install -e .
```