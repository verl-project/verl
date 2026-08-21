module load devel/miniconda

nvidia-smi

source ~/.bashrc
# some users reported issues with stacked conda environments; see https://en.wikipedia.org/wiki/Rule_of_three_(writing)
conda deactivate
conda deactivate
conda deactivate
conda activate fob

# Running the job

start=$(date +%s)

__FOB_COMMAND__

finish=$(date +%s)

runtime=$((finish-start))

echo Job execution complete.
echo Total job runtime: $runtime seconds
