# Grounded Realsense Segment Anything

Application and framework for using Grounded-SAM (Grounded-DINO + SAM) with Realsense camera(s).

## Installation
1. Install Grounded-SAM dependencies following `Grounded-Segment-Anything/README.md`
2. Install Realsense SDK 2.0 following [this guide](https://iqr-lab.github.io/computer-vision/intel-realsense.html)
3. Install this project's dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Folow the dependency installation instructions for PySide6 [here](https://pypi.org/project/PySide6/)
    > For v6.5.0: this lib must be installed:
    ```bash
    sudo apt install libxcb-cursor0
    ```
5. Download the pretrained model weights:
    ```bash
    ./download
    ```