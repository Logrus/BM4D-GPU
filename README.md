# BM4D GPU (CUDA) implementation

## About project

For details about project, please read included [PDF report](https://github.com/Logrus/BM4D-GPU/blob/master/bm4d.pdf).
## License

The majority of the code is licensed under MIT license. 

However, some header files are from NVIDIA and hold a different license, which is included at the beginning of each relevant file.


## Working with devcontainer

### Installing docker

Make sure that you have [inastalled docker](https://docs.docker.com/engine/install/) and [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Opening project in vscode

Simply open project folder in vscode and run `Dev Containers: Open Folder in Container...`.
Development environment would be build in vscode.

Alternatively, it is possible to build image using `.devcontainer/build_image.sh` script.

## Build with cmake from devcontainer

Executing from root folder of a project (where CMakeLists.txt is located).

```bash
mkdir build && \
cmake -B build -GNinja && \
cmake --build build
```
