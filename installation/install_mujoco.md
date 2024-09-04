## Install MuJoCo

```console
cd ~
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir $HOME/.mujoco
tar -xvf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
echo -e 'export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin' >> ~/.bashrc
echo -e 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
echo -e 'export PATH="$LD_LIBRARY_PATH:$PATH"' >> ~/.bashrc
```
For visualizing mujoco in a GUI viewer:
```console
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```