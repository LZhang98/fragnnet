## Build Apptainer Image

### Build Image from Apptainer def
```
mkdir apptainer
export APPTAINER_TMPDIR=$(pwd)/apptainer/tmp &&  apptainer build ./apptainer/fragnnet_cpu.sif ./fragnnet_cpu.def
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
```
apptainer shell ./fragnnet_cpu.def
pip install --user -I -e .
python
```

## Run apptainer shell
```
apptainer exec ./containers/apptainer/fragnnet_cpu.sif pip install --user -I -e .
apptainer exec ./containers/apptainer/fragnnet_cpu.sif <command>
```

apptainer shell -B ${SLURM_TMPDIR} ./containers/apptainer/fragnnet_cpu.sif 

eg. run for msgym

apptainer exec -B ${SLURM_TMPDIR} \
    ./containers/apptainer/fragnnet_cpu.sif \
    python ./scripts/ms2c/run_ms2c_predict_msgym.py \
    --device=cpu \
    --save_dp=${SLURM_TMPDIR}/msgym/fragnnet_predicted/retrieval_candidates_mass \
    --proc_dp=${SLURM_TMPDIR}/proc \
    --frag_dp=${SLURM_TMPDIR}/frag \
    --zipped_frag=True \
    --model_ckpt=./saved_ckpts/msg_fragnnet_0,01_v2_9gnx820u/msg_fragnnet_0,01_v2_9gnx820u.ckpt
