# DDPG-IL Model [(source paper)](https://ieeexplore.ieee.org/document/9164410)
...

# Training the Model
**NOTE: You need a sufficient computer to run CARLA and train this model (mainly a GPU with at least 12 GB of VRAM). Renting a VM is a good option.**

1. Clone this repository.
2. Run one of the setup scripts aligning with your operating system to install Anaconda and CARLA 0.9.15.
```
bash <path to this repository>/linux_vm_setup.sh
```
OR
```
./<path to this repository>/local_setup_windows.ps1 # perform this in PowerShell
```
3. Activate the Anaconda base environment.
```
source <path to conda>/bin/activate
```
4. Create a new conda environment for training.
```
conda create -n carla python=3.7
```
```
conda activate carla
```
5. Install the CARLA Python API using the python3 .whl file.
```
pip install <path to carla>/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl
```
6. Install the pytorch version compatible with python 3.7.
```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
7. Install the rest of the dependencies.
```
pip install -r <path to this repository>/requirements.txt
```
8. Create a .env file and add an environment variable for loading your imitation learning model for pretraining.
```
IL_MODEL="<path to your IL model>
```
9. Launch the CARLA simulator and render off-screen to optimize performance (NOTE: You need OpenGL to run CARLA). Wait a minute or so for carla to fully launch.
```
./CarlaUE4.sh -RenderOffScreen # Linux
CarlaUE4.exe  -RenderOffScreen # Windows
```
10. Run the training program, happy training!
```
python train.py
```

# Running the Trained Model
