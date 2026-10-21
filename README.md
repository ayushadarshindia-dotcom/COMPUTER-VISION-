# Drone Safety Monitor

A computer-vision program that watches a camera feed (a phone, a webcam, or a drone camera) and raises alerts when it sees:

| Alert | What it means | On screen |
|---|---|---|
| **Person down** | Someone has been lying on the ground for 5+ seconds | Yellow box (timer), then red box |
| **Fall** | Someone dropped to the ground quickly | Message in the console + snapshot |
| **Possible fight** | Two people close together with strong, sustained arm movement | Red box |
| **Group caution** | 5 or more people standing close together | Orange box |

It uses an AI model (YOLO pose) that finds 17 body points on each person, then simple rules decide what the body positions mean.

> **Important:** this is a helper, not a judge. It can only see body positions. It cannot tell if someone is unconscious, and it cannot know if a group is a dispute. A human must always check each alert.

---

## What you need

- A Windows PC or laptop (these steps use Windows PowerShell)
- Python 3.12, 3.13 or 3.14
- Internet (for installing packages and the first-time model download, about 6 MB)
- A camera: a phone with an IP-camera app, a USB webcam, or a video file
- About 2 GB of free disk space

---

## Step 1: Install Python

1. Go to **python.org** and download Python (3.12, 3.13 or 3.14).
2. Run the installer. Tick **"Add python.exe to PATH"** if you see that option.
3. Check it worked. Open PowerShell and type:
   ```powershell
   python --version
   ```
   You should see something like `Python 3.14.7`.

## Step 2: Make a project folder

1. Open PowerShell as a **normal user** (not "Run as administrator").
2. Create a folder and go into it:
   ```powershell
   cd $HOME\Documents
   mkdir drone
   cd drone
   ```
3. Put `drone_safety_monitor.py` and this `README.md` in this folder.

> Do not work inside `C:\WINDOWS\system32`. Windows blocks you from creating files there.

## Step 3: Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer.

```powershell
python -m venv drone-ai
```

## Step 4: Activate it

```powershell
drone-ai\Scripts\Activate.ps1
```

You should now see `(drone-ai)` at the start of the line.

**If you get "running scripts is disabled":** run this once, then activate again.
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
drone-ai\Scripts\Activate.ps1
```

You must activate the environment **every time** you open a new PowerShell window.

## Step 5: Install the packages

```powershell
pip install opencv-python numpy ultralytics
```

This downloads a few hundred MB (it includes PyTorch), so it can take several minutes.

## Step 6: Check the install

```powershell
python -c "import cv2, numpy, torch, ultralytics; print(cv2.__version__, torch.__version__)"
```

If it prints two version numbers, everything is installed. A message about "Ultralytics Settings" on the first run is normal.

## Step 7: Set up your camera

Pick **one** option.

### Option A: Phone as an IP camera (recommended for testing)

1. Install an IP-camera app on your phone (for example **IP Webcam** on Android).
2. Connect the phone and the PC to the **same Wi-Fi**.
3. Start the server in the app. It shows an address like `http://192.168.1.2:8080`.
4. Your video address is that address plus `/video`:
   ```
   http://192.168.1.2:8080/video
   ```
5. Check it first: paste that address into your PC's web browser. You should see live video.

Use the `http://` address, not `https://`.

### Option B: USB webcam

Use `--source 0` (try `1` or `2` if 0 does not work).

### Option C: A video file

Copy a video into the project folder and use its name, for example `--source video.mp4`.

## Step 8: Run it

```powershell
python drone_safety_monitor.py --source http://192.168.1.2:8080/video
```

- The first run downloads the AI model, so wait a moment.
- A video window opens. Press **q** in that window to quit.
- Alerts print in the console, beep (Windows), and save a snapshot in a folder called `alerts`.

Replace the address with your own.

## Step 9: Test each alert

Do these tests safely, on a soft surface.

1. **Normal:** walk around and sit on a chair. You should only see green boxes. Add `--debug` to see the numbers the program measures.
2. **Person down:** lie down on the floor in view and stay still. The box turns yellow with a counting timer, then red with `ALERT: PERSON DOWN` after 5 seconds.
3. **Fall:** have someone drop quickly onto a mat or bed. The console shows `Possible fall detected`.
4. **Fight:** two people do slow, safe play-fighting with swinging arms, close together. You should see `ALERT: POSSIBLE FIGHT`. Also test hugging, waving and dancing to see how often it triggers wrongly.
5. **Group:** have 5 or more people stand close together. You should see an orange `CAUTION` box.

---

## Options (settings you can change)

Add these after the command. Example:
```powershell
python drone_safety_monitor.py --source video.mp4 --down-seconds 8 --debug
```

| Option | Default | What it does |
|---|---|---|
| `--source` | `0` | Camera number, video address, or video file |
| `--down-seconds` | `5` | Seconds lying before the person-down alert |
| `--fall-drop` | `0.35` | How fast a drop counts as a fall (lower = more sensitive) |
| `--fight-seconds` | `2` | Seconds of aggressive movement before the fight alert |
| `--fight-speed` | `2.0` | Arm-speed threshold (higher = fewer false alarms) |
| `--fight-dist` | `1.0` | How close two people must be to count as a fight |
| `--group-size` | `5` | People needed for the group caution |
| `--skip-frames` | `0` | Run the AI on 1 of every N+1 frames (for slower computers) |
| `--imgsz` | `640` | AI input size (416 or 320 is faster, a bit less accurate) |
| `--model` | `yolov8n-pose.pt` | Which model file to use |
| `--debug` | off | Show torso angle and box shape under each person |
| `--no-beep` | off | Turn off the alert sound |
| `--no-show` | off | No video window (for a drone with no screen) |

The AI speed (`AI FPS`) shows in the corner of the video and prints in the console every 5 seconds.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Access is denied` when creating the environment | You are in `C:\WINDOWS\system32`. Go to `cd $HOME\Documents` and try again. |
| `running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `Could not open source` with a webcam | Try `--source 1` or `--source 2`. Check Windows camera privacy settings and close other apps using the camera. |
| `Could not open source` with the phone | Phone and PC must be on the same Wi-Fi. Test the address in your browser first, and use `http://` with `/video` at the end. |
| `pip` cannot find a package | Your Python is probably too new or too old. Install Python 3.12 and make the environment with it. |
| The program says `Person down` while someone is sitting | Run with `--debug` and check the numbers under the box. Make sure the camera sees the whole body, not just the upper body behind a desk. |
| Too many fight alerts | Raise `--fight-speed` (try 3.0) or `--fight-seconds` (try 3). |
| Fights are missed | Lower `--fight-speed` (try 1.5). |
| Very slow video | Use `--imgsz 416` and/or `--skip-frames 2`. |
| `ModuleNotFoundError` | The environment is not active. Run `drone-ai\Scripts\Activate.ps1` again. |

---

## Limits you should know

- The camera angle matters. Best results come from a view that shows the whole body, from low to medium height, tilted slightly down. Looking straight down from high up works poorly.
- Lying detection needs a visible torso. If the hips are hidden (behind a desk or car), the program stays silent instead of guessing.
- Fight detection is a rough guess based on arm speed. Expect false alarms from dancing, sports and play-fighting.
- It does not recognise faces or identities, and it is not designed to.
- Dim light, motion blur and a shaking camera reduce accuracy.

## Using it responsibly

- Check your local laws and rules on flying drones and on filming people before you fly or record.
- Respect people's privacy. Do not use it to follow or identify individuals.
- Do not connect it to real emergency numbers. A false alarm wastes emergency workers' time. Have a person check each alert and decide what to do.
- Test with a safety pilot and in a clear open area before any real flight.

---

## Moving to a Raspberry Pi (later)

1. Install Raspberry Pi OS (64-bit) and make a virtual environment, as in Steps 3 to 5.
2. Install with `pip install ultralytics[export]`.
3. Convert the model to the faster NCNN format:
   ```bash
   yolo export model=yolov8n-pose.pt format=ncnn imgsz=640
   ```
4. Run it with no window:
   ```bash
   python drone_safety_monitor.py --source 0 --model yolov8n-pose_ncnn_model --no-show
   ```

A Pi 5 on its own will be much slower than a laptop. For about 30 AI frames per second, you'd add an accelerator such as the Raspberry Pi AI Kit (Hailo), or use a board with a built-in AI chip, and adapt the script's AI part to it.

---

## Files in this project

| File | Purpose |
|---|---|
| `drone_safety_monitor.py` | The main program |
| `README.md` | This guide |
| `drone-ai/` | The virtual environment (created in Step 3, do not edit) |
| `alerts/` | Snapshots saved when an alert fires (created automatically) |
