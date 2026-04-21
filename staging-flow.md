# Staging flow — telearms-roboplayground

How to iterate on the Agora video render on staging without waiting for
the Agora round-trip. Intended as quick context for future sessions.

## Where things live

| Piece | Location |
|---|---|
| Local repo | `~/ruby/metal/telearms-full/telearms-roboplayground` |
| GitHub | `github.com/frodobots-org/telearms-roboplayground` |
| Working branch | `telearms-migration` |
| Staging host (SSH alias) | `thorsim-staging` |
| Staging repo path | `/home/ubuntu/telearms-roboplayground` |
| Staging venv | `/home/ubuntu/playground-venv` |
| Headless display | `:100` (Xvfb) |
| MuJoCo GL backend | `egl` |
| Stream pipeline | GStreamer → teleop_sdk → Agora RTC |

## Standard loop (push → deploy → stream test)

The server pulls from `telearms-migration` every time `update_and_run.sh`
runs, so the flow is: push → run the script on staging → check the
browser.

```bash
# 1. Commit and push to the branch
git push origin telearms-migration

# 2. Restart the sim on staging (in background, logs to /tmp/playground-run.log).
#    update_and_run.sh does: git pull + restart Xvfb + launch the sim under Telearms input mode.
ssh -o UpdateHostKeys=no thorsim-staging \
  "bash -lc 'cd /home/ubuntu/telearms-roboplayground && nohup ./update_and_run.sh > /tmp/playground-run.log 2>&1 & disown'"

# 3. Peek at the log after ~10 s
ssh -o UpdateHostKeys=no thorsim-staging "tail -30 /tmp/playground-run.log"
```

Healthy log lines look like:

```
[telearms] pushed render #150 shape=(720, 1280, 3) ... appsrc=<enum GST_FLOW_OK ...>
[telearms.ws_client][INFO] - Connected to wss://telearms-ws-server-staging.onrender.com
```

`Signal 15, shutting down` just means the previous sim was killed by the
pidfile in `update_and_run.sh` — that's normal during a redeploy.

## Fast loop — render a PNG on staging, scp it back

When iterating on camera/render changes, skip Agora entirely. Use
`tools/render_test.py` to dump one PNG per camera plus the composite the
stream would publish, then `scp` the output folder to the Mac and eyeball
it (or have Claude `Read` the PNG).

```bash
# On staging: pull latest, render every camera + the composite to /tmp/playground_render_test/
ssh -o UpdateHostKeys=no thorsim-staging \
  "cd /home/ubuntu/telearms-roboplayground && git pull --ff-only \
   && source /home/ubuntu/playground-venv/bin/activate \
   && rm -rf /tmp/playground_render_test \
   && DISPLAY=:100 python tools/render_test.py"

# Copy PNGs back to the Mac
scp -o UpdateHostKeys=no 'thorsim-staging:/tmp/playground_render_test/*.png' \
    ~/ruby/metal/telearms-full/telearms-roboplayground/tools/render_test_out/
```

The rendered files are named `cam_<id>_<name>.png` plus
`env_render_composite.png` (the actual frame the pipeline pushes to Agora).
Any path that depends on the real scene (physics, camera IDs, compiled
MuJoCo model) exercises the same code the Agora stream will use.

## What each camera ID is (RotateValve, BimanualPanda)

```
ncam = 8
 0  external                                 ← world-fixed operator cam (composite main view)
 1  head                                     ← Panda head cam (behind the robot, points wrong way)
 2  panda nohand_left/left_wrist             ← left wrist POV  (composite bottom-left overlay)
 3  panda nohand_right/right_wrist           ← right wrist POV (composite bottom-right overlay)
 4..7  handwheel_valve*_fixed/track          ← prop-local cameras, unused
```

The composite render is defined in `roboeval/roboeval_env.py::render`.
It picks `external` by name for the main view and the `*/left_wrist` /
`*/right_wrist` pair by suffix for the overlays, so it still works if
the prop cameras shift IDs.

## Tuning the external camera

`roboeval/envs/xmls/world.xml` declares the `external` camera. To change
framing, edit `pos`, `xyaxes`, and/or `fovy` there, then push + rerun the
fast loop. Reminder of MuJoCo conventions:

- `pos` is world-space meters.
- `xyaxes` gives the camera's local X axis and Y axis; -Z is where it
  looks. Don't mix units across the 6 numbers.
- `fovy` defaults to 45°. Bigger = wider = more scene = more distortion.
  Prefer moving the camera back (increase `|pos|`) over cranking `fovy`
  past ~60.

Example current values (as of last tuning pass):

```xml
<camera name="external" mode="fixed"
        pos="-0.300 -1.300 1.750"
        xyaxes="0.610 -0.793 0.000 0.289 0.222 0.931"
        fovy="55"/>
```

## Known resolution quirks

- `data_collection/base.py` passes `resolution=(900, 1000)` (width, height)
  into `TelearmsTeleop`, so the env renders 1000×900 frames (H×W).
- The GStreamer pipeline caps are 1280×720. The Telearms WebRTC wrapper
  resizes every frame before pushing — that resize slightly stretches the
  composite. It's not a bug in the composite itself; `render_test.py`
  outputs at 1280×720 so it shows the target framing without the stretch.
- `cv2` is NOT in `playground-venv` despite being in `pyproject.toml`.
  Use Pillow for image resizing in this codebase.

## Recap of recent render changes

1. `roboeval_env.py::render` now composites main + 2 wrist overlays
   (port of RoboEval@telearms), picking cameras by name.
2. `data_collection/telearms_input.py` no longer overrides `render()`
   — the base env's composite is what the pipeline sees.
3. `world.xml` external camera pulled back + `fovy=55` to de-zoom the
   stream.

## SSH host key gotcha

The first time per day you hit `thorsim-staging` you may see
`client_global_hostkeys_prove_confirm: server gave bad signature`. Pass
`-o UpdateHostKeys=no` (as in all the snippets above) to sidestep it.
