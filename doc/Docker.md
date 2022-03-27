# Stereo matching

Author: [Tobit Flatscher](https://github.com/2b-t) (January 2020)



## Docker

### 2.2 Run from Docker

This code is shipped with a [Docker](https://www.docker.com/) that allows the software to be run without having to install all the dependencies. For this one has to [set-up Docker](https://docs.docker.com/get-docker/) (select your operating system and follow the steps) as well as [Docker Compose](https://docs.docker.com/compose/install/) ideally with `$ sudo pip3 install docker-compose`.

Then browse the `docker` folder containing all the different Docker files, open a console and start the docker with

```bash
$ sudo docker-compose up
```

and then - after the container has been compiled - open another terminal and connect to the Docker

```bash
$ sudo docker-compose exec stereo_matching_docker sh
```

Now you can work inside the Docker as if it was your own machine. Later it is discussed how one can use Visual Studio Code as an IDE and not having to launch the Docker from the console.

Advantages of Docker compared to an installation on the host system are discussed in more detail [here](https://hentsu.com/docker-containers-top-7-benefits/).

When opening a Jupyter notebook from inside the container you might have to supply the following options:

```bash
$ jupyter notebook --ip=127.0.0.1 --port=8888 --allow-root
```

#### 2.2.1 Graphic user interfaces inside the Docker

Docker was actually not designed to be used with a graphic user interface. There are several workarounds for this, most of them mount relevant X11 folders from the host system into the Docker. In our case this is achieved by a corresponding Docker Compose file `docker-compose-gui.yml` that [extends](https://docs.docker.com/compose/extends/) the basic `docker-compose.yml` file.

Before launching it one has to allow the user to access the X server from within the Docker with

```bash
$ xhost +local:root
```

Then one can open the Docker by additionally supplying the command line argument `-f <filename>`

```bash
$ docker-compose -f docker-compose-gui.yml up
```

##### 2.2.1.1 Hardware accelerated OpenGL with `nvidia-container-runtime`

Another problem emerges when wanting to use hardware acceleration such as with OpenGL. In such a case one has to allow the Docker to access the host graphics card. This can be achieved with the [`nvidia-docker`](https://github.com/NVIDIA/nvidia-docker) or alternatively with the [`nvidia-container-runtime`](https://github.com/NVIDIA/nvidia-container-runtime).

Latter was chosen for this Docker: The configuration files `docker-compose-gui-nvidia.yml` and `docker-compose-nvidia.yml` inside the `docker` folder contain Docker Compose configurations for accessing the hardware accelerators inside the Docker. Former is useful when running hardware-accelerated graphic user interfaces while the latter can be used to run CUDA inside the Docker.

For this start by launching `docker info` and check if the field `runtime` additionally to the default `runc` also holds an `nvidia` runtime. If not please follow the [installation guide](https://github.com/NVIDIA/nvidia-container-runtime#installation) as well as the [engine setup](https://github.com/NVIDIA/nvidia-container-runtime#docker-engine-setup) (and then restart your computer).

Then you should be able to run the Docker Compose image with

```bash
$ docker-compose -f docker-compose-gui-nvidia.yml up
```

To verify that the hardware acceleration is actually working you can check the output of `nvidia-smi`. If working correctly it should output you the available hardware accelerators on your system.

```bash
$ nvidia-smi
```

#### 2.2.2 Docker inside Visual Studio Code

Additionally this repository comes with a Visual Studio Code project. The following sections will walk you through how this can be set-up.

##### 2.2.2.1 Set-up

If you do not have Visual Studio Code installed on your system then [install it](https://code.visualstudio.com/download). And finally follow the Docker post-installation steps given [here](https://docs.docker.com/engine/install/linux-postinstall/) so that you can run Docker without `sudo`. Finally install the [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) and [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) plugins inside Visual Studio Code and you should be ready to go.

##### 2.2.2.2 Open the project

More information about Docker with Visual Studio Code can be found [here](https://code.visualstudio.com/docs/containers/overview).

##### 2.2.2.3 Change the Docker Compose file

The Docker Compose file can be changed inside `.devcontainer/devcontainers.json`:

```json
{
  "name": "Stereo Matching Docker Compose",
  "dockerComposeFile": [
    "../docker/docker-compose.yml" // Change Docker-Compose file here
  ],
```
