from __future__ import annotations


def quantize_pong_reward(value, threshold=0.5):
  value = float(value)
  threshold = float(threshold)
  if value >= threshold:
    return 1.0
  if value <= -threshold:
    return -1.0
  return 0.0
