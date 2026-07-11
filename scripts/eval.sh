#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# W&B mirrors TensorBoard logs from runs/ when enabled.
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-ssl-lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-oc-storm}"
export WANDB_MODE="${WANDB_MODE:-online}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
SAVE_FRAMES="${SAVE_FRAMES:-false}"
SEED="${SEED:-42}"

eval_run() {
    local run_name="$1"
    local env_name="$2"
    local config_name="$3"

    echo "============================================================"
    echo "Evaluating ${run_name}"
    echo "  env: ${env_name}"
    echo "  config: ${config_name}"
    echo "  episodes: ${EVAL_EPISODES}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python -u eval.py \
        --save_frames "${SAVE_FRAMES}" \
        --eval_episodes "${EVAL_EPISODES}" \
        --run_name "${run_name}" \
        --env_name "${env_name}" \
        --seed "${SEED}" \
        --config_name "${config_name}"
}

# Default Pong comparison: oc-storm vs plain STORM.
eval_run "Pong-oc-storm" "PongNoFrameskip-v4" "atari_vector_visual"
eval_run "Pong-STORM" "PongNoFrameskip-v4" "atari_visual"

echo "============================================================"
echo "Evaluation CSVs:"
echo "  eval_results/Pong-oc-storm_episode_return.csv"
echo "  eval_results/Pong-STORM_episode_return.csv"
echo
echo "TensorBoard:"
echo "  bash scripts/tensorboard.sh"
echo "============================================================"


######################################################

# HornetProtector MageLord MantisLords HKPrime MegaMossCharger Mawlek GodTamer
# GrimmBoss BattleSisters

# eval
# boss_name="HornetProtector"
# CUDA_VISIBLE_DEVICES=0 python -u eval.py \
#     --save_frames false \
#     --eval_episodes 10 \
#     --run_name "${boss_name}-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"
