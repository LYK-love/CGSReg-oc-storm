#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# W&B mirrors TensorBoard logs from runs/ when enabled.
export WANDB_ENABLED="${WANDB_ENABLED:-1}"
export WANDB_ENTITY="${WANDB_ENTITY:-ssl-lab}"
export WANDB_PROJECT="${WANDB_PROJECT:-oc-storm}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Boxing Pong Breakout RoadRunner etc.
env_prefix="Pong"
RAM="${RAM:-False}"
SIZE_CONFIG="${SIZE_CONFIG:-${STORM_SIZE_CONFIG:-base}}"
TRAIN_RAM_ARGS=()
if [[ "${RAM}" == "True" || "${RAM}" == "true" || "${RAM}" == "1" ]]; then
    TRAIN_RAM_ARGS+=(--ram)
fi
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --run_name "${env_prefix}-DEV" \
    --env_name "${env_prefix}NoFrameskip-v4" \
    --seed 42 \
    --config_name "atari_vector_visual" \
    --size-config "${SIZE_CONFIG}" \
    "${TRAIN_RAM_ARGS[@]}" \
    --save_eval_dataset "${SAVE_EVAL_DATASET:-True}" \
    --eval_every_steps "${EVAL_EVERY_STEPS:-20000}" \
    --eval_metrics_batches "${EVAL_METRICS_BATCHES:-1}"

######################################################

# HornetProtector MageLord MantisLords HKPrime MegaMossCharger Mawlek GodTamer
# GrimmBoss BattleSisters

# boss_name="HornetProtector"
# CUDA_VISIBLE_DEVICES=0 python -u train.py \
#     --run_name "${boss_name}-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"

# boss_name="MantisLords"
# python -u train_async.py \
#     --run_name "${boss_name}-async-DEV" \
#     --env_name "HollowKnight/${boss_name}" \
#     --seed 42 \
#     --config_name "hollow_knight_vector_visual"
