# BM4D GPU (CUDA) implementation

## About project

TODO: write summary

## License

The majority of the code is licensed under MIT license. 

However, some header files are from NVIDIA and hold a different license, which is included in the header.


## Working with devcontainer

### Installing docker

TODO: provide pointers to install docker, also note on nvidia package

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