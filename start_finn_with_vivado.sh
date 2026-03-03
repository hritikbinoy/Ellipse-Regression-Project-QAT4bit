#!/bin/bash
# Start FINN Docker with Vivado 2023.2 using run-docker.sh

echo "Starting FINN Docker with Vivado 2023.2..."
echo "Vivado will be mounted from /tools/Xilinx/Vivado/2023.2"
echo ""

cd /home/hritik/Desktop/Hritik/Project/ellipse-regression-project/finn_repo

# Point to your host Xilinx installation
export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2023.2

# Optional: Set other Xilinx paths explicitly (not required but sometimes helpful)
export VIVADO_PATH=$FINN_XILINX_PATH/Vivado/$FINN_XILINX_VERSION
export VITIS_PATH=$FINN_XILINX_PATH/Vitis/$FINN_XILINX_VERSION
export HLS_PATH=$FINN_XILINX_PATH/Vitis_HLS/$FINN_XILINX_VERSION

# Use existing Docker image (skip rebuild)
export FINN_DOCKER_PREBUILT=0
export FINN_DOCKER_TAG="finn:local-vivado2023.2"

# Skip XRT download (already set as default in run-docker.sh now)
export FINN_SKIP_XRT_DOWNLOAD=1

./run-docker.sh notebook

echo ""
echo "Jupyter stopped. To restart, run this script again."
