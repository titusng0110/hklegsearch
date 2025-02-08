eval "$(conda shell.bash hook)"
conda activate /mnt/nvme/lmao
python -m gunicorn -c gunicorn.conf.py
