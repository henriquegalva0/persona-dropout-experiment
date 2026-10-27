#!/bin/bash
#SBATCH --job-name=vllm-persona
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_%a.out
# Servidor vLLM do perfil `cluster`: 1 modelo por GPU, porta vllm.base_port + índice (10980–10985).
# NÃO testado (escrito no ambiente local). Depois de subir: PROFILE=cluster python src/checks.py v3 v4 v9
#
# Se o sbatch falhar com "I/O error writing script/environment to file" (problema do controlador):
#   salloc --gres=gpu:1 ...; tmux; SLURM_ARRAY_TASK_ID=<i> bash slurm/vllm_array.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

# Imagem vLLM geral com versão fixada, puxada uma vez:
#   apptainer pull vllm.sif docker://vllm/vllm-openai:<versão>   (registre a versão no manifest)
SIF=${SIF:?defina SIF=/caminho/para/vllm.sif}
i=${SLURM_ARRAY_TASK_ID:?defina SLURM_ARRAY_TASK_ID}

eval "$(apptainer exec "$SIF" python3 -c "
import yaml
c = yaml.safe_load(open('config.yaml')); m = c['models'][$i]; v = c['vllm']
print(f\"MODEL={m['id']} REV={m['revision']} PORT={v['base_port'] + $i} MAXLEN={v['max_model_len']}\")")"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   # max_position_embeddings=4096 no config; avaliamos até 64K
apptainer exec --nv "$SIF" vllm serve "$MODEL" \
  --revision "$REV" \
  --served-model-name "${MODEL#*/}" \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --dtype bfloat16
