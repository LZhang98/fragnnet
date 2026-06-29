echo 'make sure run this at top level project dir'
echo 'this will require wandb, python3.11 and stdenv/2023 module loaded'

# Args forwarded to scripts/misc/download_ckpts.py
INPUT_FILE=${INPUT_FILE:-"./data/wandb_downloads/deepmet_runs.json"}
SAVE_DIR=${SAVE_DIR:-"./saved_ckpts/"}
USE_CACHED=${USE_CACHED:-0} # set to 1 to enable --use_cached

EXTRA_ARGS="--input_file ${INPUT_FILE} --save_dir ${SAVE_DIR}"
if [ "${USE_CACHED}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_cached"
fi

if [ ! -d "wandb_venv" ]; then
    echo "Virtual environment wandb_venv does not exists"
    echo "Creating virtual environment..."
    virtualenv wandb_venv
    source wandb_venv/bin/activate
    pip install --no-index --upgrade pip
    pip install --no-index wandb
    python scripts/misc/download_ckpts.py ${EXTRA_ARGS}
else
    echo "Virtual environment wandb_venv already exists"
    source wandb_venv/bin/activate
    python scripts/misc/download_ckpts.py ${EXTRA_ARGS}
fi


