cd ~/Downloads/bubble_tracking/third_party/opencv
rm -rf build
mkdir build && cd build

# Ensure CUDA 12.8 is used
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# Make sure venv is active
python -c "import sys; print(sys.executable)"

PY_EXE="$(python -c 'import sys; print(sys.executable)')"
PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"

cmake -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="$PY_SITE" \
  -D OPENCV_PYTHON3_INSTALL_PATH="$PY_SITE" \
  -D CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
  -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D WITH_CUDA=ON \
  -D CUDA_ARCH_BIN=8.9 \
  -D WITH_CUBLAS=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_opencv_python3=ON \
  -D PYTHON3_EXECUTABLE="$PY_EXE" \
  ..

ninja -j"$(nproc)"
ninja install