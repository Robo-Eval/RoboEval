
## Installation
```
git clone --recurse-submodules git@github.com:frodobots-org/RoboEval.git
cd RoboEval
# Create and activate conda environment:
conda create -n roboeval python=3.10
conda activate roboeval
pip install -e .
pip install -e ".[examples]"
sudo apt install -y libgirepository1.0-dev
cd telearms
pip install -r requirements.txt
pip install ./teleop_sdk-0.1-py3-none-any.whl
```

## How to run telearms version
```
source <your conda path>/miniconda3/bin/activate
conda activate roboeval
export PYTHONPATH=<your repo path>/RoboEval:$PYTHONPATH

# if host on the EC2, please uncomment the following lines to enable GPU rendering
# export DISPLAY=:99
# Xvfb :99 -screen 0 1024x768x24 &
# export MUJOCO_GL=egl

# this version will connect to a websocket server to asign channel. you can try with test server.
# python telearms/test_server.py

cd roboeval
export SECRET_ID=<your secret id>
export SECRET_KEY=<your secret key>
python data_collection/demo_recorder.py input_mode=Telearms robot="Bimanual Panda" env="Empty Environment"
```


### Error Code
```
ERR_INVALID_APP_ID = 101
ERR_INVALID_CHANNEL_NAME = 102
ERR_NO_SERVER_RESOURCES = 103
ERR_LOOKUP_CHANNEL_REJECTED = 105
ERR_OPEN_CHANNEL_REJECTED = 107
ERR_TOKEN_EXPIRED = 109
ERR_INVALID_TOKEN = 110
ERR_DYNAMIC_TOKEN_BUT_USE_STATIC_KEY = 115
ERR_SET_CLIENT_ROLE_NOT_AUTHORIZED = 119
ERR_DECRYPTION_FAILED = 120
ERR_OPEN_CHANNEL_INVALID_TICKET = 121
ERR_OPEN_CHANNEL_TRY_NEXT_VOS = 122
ERR_CLIENT_IS_BANNED_BY_SERVER = 123
```
