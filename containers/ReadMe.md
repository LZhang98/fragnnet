## Build Apptainer Image

### Build Image from Apptainer def
CPU Image
```
export APPTAINER_TMPDIR=$(pwd)/tmp &&  apptainer build ./fragnnet_cpu.sif ./fragnnet_cpu.def
```

CUDA Image
```
export APPTAINER_TMPDIR=$(pwd)/tmp &&  apptainer build ./fragnnet_cuda118.sif ./fragnnet_cuda118.def
```

### Get Prebuild image from google drive
CPU Image
```
mkdir -p apptainer
cd apptainer
rclone copy -P fragnnet:containers/apptainer/fragnnet_cpu.sif ./apptainer/
```

CUDA Image
```
mkdir -p apptainer
cd apptainer
rclone copy -P fragnnet:containers/apptainer/fragnnet_cuda118.sif ./apptainer/
```

## Useful links
* https://apptainer.org/docs/user/main/build_a_container.html
* https://apptainer.org/docs/user/main/build_a_container.html#building-containers-from-apptainer-definition-files
* https://apptainer.org/docs/user/main/appendix.html#id10
* https://apptainer.org/docs/user/main/docker_and_oci.html#containers-cached-by-the-docker-daemon
* https://apptainer.org/docs/user/main/persistent_overlays.html


### Run apptainer on CC
```
module load apptainer/1.2.4

```
## Run apptainer shell
At Project Root
```
apptainer shell ./containers/fragnnet_cpu.sif
pip install --user -I -e .
python
```

## Run apptainer shell
do this 
```
apptainer exec ./containers/fragnnet_cpu.sif <command>
```
Check install package
```
apptainer exec ./containers/fragnnet_cpu.sif python -m pip list
```

Check install package
```
apptainer exec ./containers/fragnnet_cpu.sif python -c "import numpy; print(numpy.__version__)"
```

install fragnnet from src
```
apptainer exec ./containers/fragnnet_cpu.sif pip install -I -e .
```
Note: When the Host Home Directory is Bound:
    By default, Apptainer often binds the host’s home directory to the container’s home directory. In this case: When you run pip install --user inside the container, it will install the Python packages in your host’s home directory under ~/.local/lib/pythonX.X/site-packages, where X.X corresponds to the Python version.
When the Container’s Home Directory is Isolated:
    If you prevent Apptainer from binding the host’s home directory by using the --no-home option or setting a custom --home path inside the container:
    The --user flag will install the packages in the container’s internal user home directory (e.g., /root/.local/lib/pythonX.X/site-packages or /home/user/.local/lib/pythonX.X/site-packages), depending on the user inside the



apptainer shell -B ${SLURM_TMPDIR} ./containers/fragnnet_cpu.sif 

eg. run for msgym

apptainer exec -B ${SLURM_TMPDIR} \
    ./containers/fragnnet_cpu.sif \
    python ./scripts/ms2c/run_ms2c_predict_msgym.py \
    --device=cpu \
    --save_dp=${SLURM_TMPDIR}/msgym/fragnnet_predicted/retrieval_candidates_mass \
    --proc_dp=${SLURM_TMPDIR}/proc \
    --frag_dp=${SLURM_TMPDIR}/frag \
    --zipped_frag=True \
    --model_ckpt=./saved_ckpts/msg_fragnnet_0,01_v2_9gnx820u/msg_fragnnet_0,01_v2_9gnx820u.ckpt
