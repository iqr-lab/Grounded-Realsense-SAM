from PySide6.QtCore import QThread, Signal
import numpy as np
import cv2

from grounded_light_hqsam import GroundedLightHQSAM
import time


class Camera(QThread):
    colorFramesReady = Signal(np.ndarray)
    depthFramesReady = Signal(np.ndarray)

    def __init__(self, color_q, depth_q):
        QThread.__init__(self)
        self.color_q = color_q
        self.depth_q = depth_q
        self.prompt = ""
        self.stop = True
        print("Camera QThread Opening...")

        # Setup Grounded-SAM Model
        print("Setting up Grounded-SAM Model...")
        self.model = GroundedLightHQSAM()
        print("Grounded-SAM Model Setup Complete")

    def run(self):
        self.keepRunning = True

        while self.keepRunning:
            start = time.time()
            if not self.stop:
                transformed_frame = self.model.predict(
                    cv2.flip(cv2.cvtColor(self.color_q.get(), cv2.COLOR_BGR2RGB), -1),
                    self.prompt,
                    box_threshold=0.4,
                    nms_threshold=0.8,
                )
                print(f"Processing averages {1/(time.time() - start)} fps")
                self.colorFramesReady.emit(transformed_frame)
            else:
                self.colorFramesReady.emit(
                    cv2.flip(cv2.cvtColor(self.color_q.get(), cv2.COLOR_BGR2RGB), -1)
                )
            self.depthFramesReady.emit(
                cv2.flip(cv2.cvtColor(self.depth_q.get(), cv2.COLOR_BGR2RGB), -1)
            )
        print("Camera QThread Closed")

    def stop_prompt(self):
        self.stop = True

    def set_prompt(self, prompt):
        self.prompt = prompt
        self.stop = False

    def close(self):
        print("Camera QThread Closing...")
        self.keepRunning = False
