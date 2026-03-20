# Repo for Synthetic CoT Data

All scripts for training and evaluation are in `/scripts`. Please modify `PROJECT_DIR` in scripts to set the directory for storing model checkpoints and logs. Usage of the scripts are instructed upon running `bash <script_name.sh>` without any parameters.

Before any training / evaluating, please modify and run `fake_hf_cache.sh` to load the model of your desire into HF cache.

Currently, training uses LoRA to reduce GPU usage. Please run the python scripts for model merging after training.