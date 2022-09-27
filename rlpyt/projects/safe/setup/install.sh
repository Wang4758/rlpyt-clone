#!/bin/bash

CONDA_ENV_NAME=$1
CP_MTH_INSTALL_DIR="$(pwd)"

echo "Installing in ${CP_MTH_INSTALL_DIR}"
echo "Conda env name is $1"

if conda env list | grep -q /envs/${CONDA_ENV_NAME}$; then
	echo "Conda env ${CONDA_ENV_NAME} already exists."
	exit
fi

# echo "Installing libraries, if you dont want that just ctrl-c"
# sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf

# 0. Install commonroad as well
if [ ! -d "$CP_MTH_INSTALL_DIR/commonroad-rl" ]; then
	cd ${CP_MTH_INSTALL_DIR}
	git clone -b safety-mtcp git@gitlab.lrz.de:ss20-mpfav-rl/commonroad-rl.git
	cd commonroad-rl
	git submodule update --init

	cd ${CP_MTH_INSTALL_DIR}/commonroad-rl
	# sed -i '$ d' scripts/install.sh
	conda env create --name ${CONDA_ENV_NAME} -f environment.yml
	source activate ${CONDA_ENV_NAME}
	bash scripts/install.sh -e ${CONDA_ENV_NAME} --no-root
else
	echo "Skipping commonroad-rl, already exists"
fi


# 1. mujoco binaries (python package installs with safety_gym)
MUJOCO_DIR="/home/$(whoami)/.mujoco/mujoco200"
if [ ! -d "$MUJOCO_DIR" ]; then
    mkdir -p ~/.mujoco && cd ~/.mujoco
    wget -N https://roboti.us/download/mujoco200_linux.zip
    unzip -o mujoco200_linux.zip
    mv mujoco200_linux mujoco200
    rm mujoco200_linux.zip
    wget -N https://roboti.us/file/mjkey.txt
else
    echo "mujoco200 folder exists, skipping installation"
fi

# 2. 
cd ${CP_MTH_INSTALL_DIR}
git clone https://github.com/openai/safety-gym.git --depth 1
cd safety-gym
# remove numpy (BUT NEED TO KEEP THE gym VERSION, otherwise safetygym give error with observation!)
sed -i '/numpy/ d' setup.py
pip install -e .

# 2.5 cr_monitor

conda install conda-build -y

cd ${CP_MTH_INSTALL_DIR}
git clone git@gitlab.lrz.de:ga58jol/cr_monitor_gym.git -b cmdp-and-extend-obs

cd cr_monitor_gym/
git submodule init
git submodule update --recursive

cd external/stl_crmonitor/
conda env update --name ${CONDA_ENV_NAME} --file environment.yml
git submodule update --init --recursive
pip install external/rtamt
conda develop .
cd crmonitor/tests
python -m unittest

cd ${CP_MTH_INSTALL_DIR}
cd cr_monitor_gym/
pip install -e .

# 3. rlpyt and deps
cd ${CP_MTH_INSTALL_DIR}
cd rlpyt
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -e .

# 3.1 additional deps (for rlpyt) and fix numpy version
pip install --upgrade numpy==1.21
pip install psutil pyprind

# 4. install starter agents
cd ${CP_MTH_INSTALL_DIR}
git clone git@gitlab.lrz.de:ga84moc/safety-starter-agents-clone.git safety-starter-agents
cd safety-starter-agents
pip install -e .

cd ${CP_MTH_INSTALL_DIR}
echo "INSTALLATION DONE"
echo "Setup pickles into the commonroad-rl directory, then run scripts/train/train_dcppo.py to see if it works..."
