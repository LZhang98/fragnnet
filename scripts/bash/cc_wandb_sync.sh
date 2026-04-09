echo 'make sure run this at top level project dir'
echo 'this will require wandb, python3.11 and stdenv/2023 module loaded'
if [ ! -d "wandb_venv" ]; then
    echo "Virtual environment wandb_venv does not exists"
    echo "Creating virtual environment..."
    virtualenv wandb_venv
    source wandb_venv/bin/activate
    pip install --no-index --upgrade pip
    pip install --no-index wandb==0.21.2

    # https://github.com/wandb/wandb/issues/2423
    wandb sync --sync-all --append
else
    echo "Virtual environment wandb_venv already exists"
    source wandb_venv/bin/activate
    # https://github.com/wandb/wandb/issues/2423
    wandb sync --sync-all --append
fi


