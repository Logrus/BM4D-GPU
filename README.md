# BM4D GPU (CUDA) implementation

## License

The majority of the code is licensed under MIT license. 

However, some header files are from NVIDIA and hold a different license, which is included in the header.


## Working with devcontainer

### Installing docker

Make sure that you have [installed docker](https://docs.docker.com/engine/install/) and [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Then run script to build image locally using `.devcontainer/build_image.sh` script.

```bash
bash .devcontainer/build_image.sh
```

It is necessary that your CUDA SDK version on host machine is matching the one inside container.
For that, different base images could be picked up and `build_image.sh` script tries to infer
CUDA SDK version automatically and pick up corresponding nvidia/cuda image from dockerhub.
In case this doesn't work for some reason, you might be forced to modify `Dockerfile` manually and pick up corresponding base image.

You can check your CUDA SDK version by running `nvidia-smi`

![check cuda sdk version via nvidia-smi](docs/images/nvidia_version.png)

In this example, cuda version is 12.2 and corresponding docker image would be `nvidia/cuda:12.2.0-devel-ubuntu22.04`. Note, even though the displayed version in `nvidia-smi` 12.2 (without last zero), the docker image version is 12.2.0 (with zero).

### Opening project in vscode

After image is built, simply open project folder in vscode and run `Dev Containers: Open Folder in Container...`.

## Build with cmake

Executing from root folder of a project (where CMakeLists.txt is located).

```bash
cmake -B build -GNinja && \
cmake --build build
```

Alternatively, you can simply run convenience `build.sh` script.

```bash
./build.sh
```

## Building and running tests

In order to build and run tests you can do:

```bash
cmake -B build -GNinja -DBUILD_TESTS=ON && \
cmake --build build && \
ctest --test-dir build/tests
```

Alternatively, you can simply run convenience script:

```bash
./tests_run.sh
```