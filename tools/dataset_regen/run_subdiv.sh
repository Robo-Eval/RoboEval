#!/usr/bin/env bash
S=/tmp/claude-0/-weka-robots-default-helenw-ICRA2026/ee5dec8d-66de-4c8d-ae9c-69e724ed7cbc/scratchpad
PY=/root/rbenv/bin/python
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 PYTHONPATH=/root/RoboEval-git HF_HOME=/root/hf
cd /root/RoboEval-git
VARS="LiftPotPosition LiftTrayPositionAndOrientation StackTwoBlocks StackTwoBlocksPositionAndOrientation
PackBoxOrientation PackBox LiftTrayPosition RotateValvePositionAndOrientation RotateValve
PickSingleBookFromTable PickSingleBookFromTablePosition LiftPotOrientation PackBoxPosition
PackBoxPositionAndOrientation RotateValvePosition CubeHandoverPosition CubeHandoverOrientation
CubeHandoverPositionAndOrientation CubeHandover LiftPot LiftTrayOrientation
PickSingleBookFromTablePositionAndOrientation StackTwoBlocksPosition StackSingleBookShelf
LiftPotPositionAndOrientation StackSingleBookShelfPosition LiftTray
PickSingleBookFromTableOrientation StackSingleBookShelfPositionAndOrientation StackTwoBlocksOrientation"
n=0
for v in $VARS; do
  for i in 0 1; do
    MUJOCO_EGL_DEVICE_ID=$i $PY "$S/render_subdiv.py" "$v" "roboeval_sd_p$i" \
      "$S/render2/${v}_p$i.txt" -1 "$i" 2 > "$S/render2/${v}_p$i.log" 2>&1 &
    n=$((n+1))
  done
done
echo "launched $n workers (2 shards/variation, 1 per GPU)"
wait
echo "SUBDIV_RENDER_DONE"
