#!/bin/bash
# Credits: https://github.com/swiss-ai/code-gym/tree/main
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --container-writable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus=0
#SBATCH --time=12:00:00
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

USERNAME="$(whoami)"
PORT="${PORT:-8000}"
CODE_GYM_DIR="${CODE_GYM_DIR:-/users/${USERNAME}/scratch/code-gym}"

# Native mode runs user snippets in lightweight enroot workers behind the
# scheduler. Keep these values aligned with the CPU count requested above.
SANDBOX_MODE="${SANDBOX_MODE:-native}"
SANDBOX_WORKERS="${SANDBOX_WORKERS:-130}"
SANDBOX_CPUS_PER_WORKER="${SANDBOX_CPUS_PER_WORKER:-1}"
SANDBOX_BASE_PORT="${SANDBOX_BASE_PORT:-11990}"
NATIVE_SQSH_PATH="${NATIVE_SQSH_PATH:-/capstor/store/cscs/swissai/infra01/reasoning/data/codegym/python+3.11-slim.sqsh}"
SANDBOX_SQSH_PATH="${SANDBOX_SQSH_PATH:-/capstor/store/cscs/swissai/infra01/reasoning/imgs/sandbox/sandbox.sqsh}"

echo "CODE_GYM_DIR=${CODE_GYM_DIR}"
echo "PORT=${PORT}"
echo "SANDBOX_MODE=${SANDBOX_MODE}"
echo "SANDBOX_WORKERS=${SANDBOX_WORKERS}"
echo "SANDBOX_CPUS_PER_WORKER=${SANDBOX_CPUS_PER_WORKER}"
echo "SANDBOX_BASE_PORT=${SANDBOX_BASE_PORT}"

[[ -d "${CODE_GYM_DIR}" ]] || { echo "Missing CODE_GYM_DIR=${CODE_GYM_DIR}" >&2; exit 1; }
[[ -f "${CODE_GYM_DIR}/src/backend/scheduler.py" ]] || { echo "Missing code-gym scheduler.py" >&2; exit 1; }

export ENROOT_DATA_PATH="/tmp/${USER}/enroot_data_${SLURM_JOB_ID}"
mkdir -p "${ENROOT_DATA_PATH}"

if [[ "${SANDBOX_MODE}" == "native" ]]; then
  SQSH_PATH="${NATIVE_SQSH_PATH}"
  CONTAINER_NAME="python-slim"
else
  SQSH_PATH="${SANDBOX_SQSH_PATH}"
  CONTAINER_NAME="sandbox"
fi
[[ -f "${SQSH_PATH}" ]] || { echo "SQSH image not found: ${SQSH_PATH}" >&2; exit 1; }

cleanup() {
  set +e
  enroot remove -f "${CONTAINER_NAME}" >/dev/null 2>&1
  rm -rf "${ENROOT_DATA_PATH}"
}
trap cleanup EXIT

if enroot list | grep -qx "${CONTAINER_NAME}"; then
  echo "Container ${CONTAINER_NAME} already exists."
else
  enroot create --name "${CONTAINER_NAME}" "${SQSH_PATH}"
fi

cd "${CODE_GYM_DIR}"

UV_BIN="${HOME}/.local/bin/uv"
if [ ! -x "${UV_BIN}" ]; then
  mkdir -p "${HOME}/.local/bin"
  curl -fsSL https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

uv python install 3.12.12
if [ ! -d ".venv" ]; then
  uv venv --python 3.12.12 .venv
fi
source .venv/bin/activate

pyv="$(python -c 'import sys; print(sys.version.split()[0])')"
[ "${pyv}" = "3.12.12" ] || { echo "Got Python ${pyv}, expected 3.12.12" >&2; exit 1; }

if [ -f requirements.txt ]; then
  uv pip sync requirements.txt
fi

exec python src/backend/scheduler.py \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --mode "${SANDBOX_MODE}" \
  --workers "${SANDBOX_WORKERS}" \
  --base-port "${SANDBOX_BASE_PORT}" \
  --cpus "${SANDBOX_CPUS_PER_WORKER}"
