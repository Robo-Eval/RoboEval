# Data Collection

## Supported Methods
Two methods of data collection are supported: **Keyboard** and **VR**.

### Keyboard
To start data collection using the keyboard, run the following command:
```sh
python PATH_TO_ROBOEVAL/roboeval/data_collection/demo_recorder.py
```

#### Controls
- **WASDZC** → Move the left arm
- **JIKLOU** → Move the right arm
- **T** → Toggle between position and rotation modes
- **R** → Start recording
- **X** → Save recording
- **ESC + Q** → Quit

#### Important Note on Keyboard Controls

**Keyboard Conflict Management**: Some keyboard controls overlap with MuJoCo's built-in GUI functions. If you notice unexpected GUI elements appearing during operation, pressing the same key again will deactivate the corresponding MuJoCo interface feature.

### VR Teleoperation with Oculus Quest

This guide provides step-by-step instructions to set up and use the Oculus Quest for VR teleoperation. Follow these steps to install necessary tools, configure your device, and test the teleoperation module.

---

#### Prerequisites

##### Set Up Git LFS (Large File Storage)
Run the following commands to install and configure Git LFS:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install  # Run this only once per user account
```

##### Clone the Oculus Reader Repository
Download the required Oculus Reader tool:

```bash
git clone https://github.com/helen9975/oculus_reader.git
```

##### Install ADB (Android Debug Bridge)
ADB is required for communication with the Oculus device:

```bash
sudo apt install android-tools-adb
```

---

#### Setting Up Your Oculus Quest

##### 1. Get Your Oculus Quest Account Name
If you haven't used Oculus Quest before, start the headset and complete the setup process, including creating a profile.

##### 2. Enable Developer Mode
Developer mode is necessary to interface with your computer.

- **If you're part of PRL**, ask Helen for the organization account.
- **Otherwise**, create your own organization at [Oculus Developer Portal](https://developer.oculus.com/manage/organizations/create/) and follow the registration steps.

##### Enable Developer Mode on Your Device
1. Turn on your Oculus Quest.
2. Open the **Oculus app** on your phone.
3. Go to **Settings** and select your device.
4. Navigate to **More Settings > Developer Mode**.
5. Enable the **Developer Mode** toggle.
6. Connect your Oculus to your computer using a **USB-C cable**.
7. Wear the device and **allow USB debugging** when prompted. Select **Always allow from this computer** to prevent repeated prompts.

---

#### Verify USB Communication
After enabling Developer Mode and connecting via USB, verify that the Oculus device can communicate with your computer:

```bash
python oculus_reader/reader.py
```
This script should display data from both the **left and right controllers**.

---

#### Testing the Oculus Teleop Module
To test teleoperation functionality, run:

```bash
python PATH_TO_ROBOEVAL/roboeval/data_collection/demo_recorder.py --input_mode Oculus
```

📌 **Note:** We use **Hydra** for configuration management. Additional parameters can be found in `roboeval/configs/data_collection.yaml`.

---

#### VR Teleoperation Controls

| **Control**                          | **Action** |
|--------------------------------------|-----------|
| **Right controller grip button**     | Hold to enable robot movement |
| **Left joystick**                    | Move in the X (left/right) and Y (forward/backward) directions |
| **Right joystick**                   | Move in the Z (up/down) direction and rotate (clockwise/counterclockwise) |
| **Right trigger button**             | Hold to close the gripper, release to open |
| **Releasing & regripping (Right grip)** | Re-calibrates the origin |
| **Button A**                         | Start recording |
| **Button B**                         | Save recording |

---

#### Troubleshooting
If you encounter issues:
- Ensure **Developer Mode** is enabled.
- Confirm that **USB Debugging** is allowed.
- Run `adb devices` to verify that the device is recognized.
- Restart your computer and try again.
