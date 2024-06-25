## Build Apptainer Image

### Build Image from Apptainer def
```
export APPTAINER_TMPDIR=$(pwd)/apptainer_images/tmp &&  apptainer  build ./apptainer_images/fragnnet_cpu.sif fragnnet_cpu.def
```

## Useful links
* https://apptainer.org/docs/user/main/build_a_container.html
* https://apptainer.org/docs/user/main/build_a_container.html#building-containers-from-apptainer-definition-files
* https://apptainer.org/docs/user/main/appendix.html#id10
* https://apptainer.org/docs/user/main/docker_and_oci.html#containers-cached-by-the-docker-daemon
* https://apptainer.org/docs/user/main/persistent_overlays.html

### Run apptainer
