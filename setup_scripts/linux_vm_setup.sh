#!/bin/bash

# Define variables
ANACONDA_FILE="Anaconda3-2024.06-1-Linux-x86_64.sh"
CARLA_VERSION="0.9.15"
CARLA_FILE="CARLA_${CARLA_VERSION}.tar.gz"
CARLA_URL="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/${CARLA_FILE}"
CARLA_DIR="/opt/carla-simulator"
# GITHUB_REPO="https://github.com/ENDEAVR-E2E-Autonomous-Driving/DDPG-IL"

# Update and install necessary system dependencies
sudo apt-get update
sudo apt-get -y install libomp5 wget tar

# Download and install Anaconda
wget https://repo.anaconda.com/archive/${ANACONDA_FILE}
bash ${ANACONDA_FILE} -b -p $HOME/anaconda3
export PATH="$HOME/anaconda3/bin:$PATH"

# Download CARLA
wget ${CARLA_URL}

# Unpack CARLA
sudo mkdir -p ${CARLA_DIR}
sudo tar -xzvf ${CARLA_FILE} -C ${CARLA_DIR}

# Install CARLA Python module and dependencies
# python -m pip install carla==${CARLA_VERSION}
# python -m pip install -r ${CARLA_DIR}/PythonAPI/examples/requirements.txt

# Clone the GitHub repository
# git clone ${GITHUB_REPO}

# Clean up
rm ${ANACONDA_FILE} ${CARLA_FILE}

echo "Installation completed successfully!"
