#!/usr/bin/env bash
export HF_HOME=/root/hf
B=/root/hf/lerobot
cd /root/RoboEval-git
merge() {
  local v=$1 pre=$2
  local ids="["
  for i in 0 1 2 3 4 5; do [ $i -gt 0 ] && ids+=", "; ids+="'$B/roboeval_ee_delta_20hz_scaled_${pre}$i/$v'"; done
  ids+="]"
  rm -rf "$B/roboeval_ee_delta_20hz_scaled/$v"
  /root/rbenv/bin/lerobot-edit-dataset \
    --repo_id "$B/roboeval_ee_delta_20hz_scaled/$v" \
    --operation.type merge --operation.repo_ids "$ids" > "$S_LOG/$v.merge.log" 2>&1 \
    && echo "OK $v" || echo "FAIL $v"
}
export -f merge
S_LOG=/tmp/claude-0/-weka-robots-default-helenw-ICRA2026/ee5dec8d-66de-4c8d-ae9c-69e724ed7cbc/scratchpad/render
export S_LOG B
for spec in "PackBox:s" "StackTwoBlocks:s" "LiftTrayPositionAndOrientation:s" \
            "StackTwoBlocksPositionAndOrientation:g1s" "PackBoxOrientation:g1s" \
            "RotateValvePosition:g2s" "RotateValvePositionAndOrientation:g2s"; do
  merge "${spec%%:*}" "${spec#*:}" &
done
wait
echo "MERGE_ALL_DONE"
