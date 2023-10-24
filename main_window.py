import numpy as np
import qimage2ndarray
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera import Camera


class MainWindow(QMainWindow):
    def __init__(self, color_q, depth_q, app):
        QMainWindow.__init__(self)
        self.setWindowTitle("IQR Lab Stream")

        self.widget = QWidget()

        # Camera Components
        self.color_label = QLabel(self.widget)
        self.depth_label = QLabel(self.widget)

        # Prompt Components
        self.prompt_label = QLineEdit(self.widget)
        self.prompt_label.setPlaceholderText("Prompt (Press Enter to submit)")
        self.prompt = ""
        self.stop_button = QPushButton("Stop", self.widget)
        self.stop = True

        self.active_prompt = QLabel()
        self.active_prompt.setText("No active prompt.")

        # Styling
        self.color_label.setStyleSheet("margin-bottom: 100px;")
        self.depth_label.setStyleSheet("margin-bottom: 100px;")
        self.prompt_label.setStyleSheet(
            "margin-bottom :80px; margin-left :80px; border: 2px solid gray; border-radius: 10px; padding: 2 8px;"
        )
        self.active_prompt.setStyleSheet("margin-left :80px;")
        self.stop_button.setStyleSheet(
            "margin-bottom :80px; margin-right :80px; margin-top: 25px; padding :5px;"
        )

        # Layouts
        self.parentWidgetLayout = QVBoxLayout()
        self.cameraWidgetLayout = QHBoxLayout()
        self.promptWidgetLayout = QHBoxLayout()
        self.activePromptWidgetLayout = QVBoxLayout()

        # Active Prompt Layout
        self.activePromptWidgetLayout.addWidget(self.active_prompt)
        self.activePromptWidgetLayout.addWidget(self.prompt_label)

        # Camera Layout
        self.cameraWidgetLayout.addWidget(self.color_label)
        self.cameraWidgetLayout.addWidget(self.depth_label)

        # Prompt Layout
        self.promptWidgetLayout.addLayout(self.activePromptWidgetLayout)
        self.promptWidgetLayout.addWidget(self.stop_button)

        self.parentWidgetLayout.addLayout(self.cameraWidgetLayout)
        self.parentWidgetLayout.addLayout(self.promptWidgetLayout)

        self.widget.setLayout(self.parentWidgetLayout)

        self.setCentralWidget(self.widget)

        # Camera Connection
        self.camera = Camera(color_q, depth_q)
        self.camera.colorFramesReady.connect(self.StartColorCameraUpdates)
        self.camera.depthFramesReady.connect(self.StartDepthCameraUpdates)
        self.camera.start()

        # Prompt Connection
        self.stop_button.clicked.connect(self.stop_button_clicked)
        self.prompt_label.returnPressed.connect(self.return_pressed)

        # Cleanup Camera Thread
        app.aboutToQuit.connect(self.camera.close)

    def stop_button_clicked(self):
        self.stop = True
        self.active_prompt.setText("No active prompt.")
        self.camera.stop_prompt()

    def return_pressed(self):
        self.stop = False
        self.prompt = self.prompt_label.text()
        self.prompt_label.setText("")
        self.active_prompt.setText(f"Active prompt: {self.prompt}")
        self.camera.set_prompt(self.prompt)

    def realsenseFrameToQImage(self, frame):
        result = qimage2ndarray.array2qimage(frame)
        return result

    @Slot(np.ndarray)
    def StartColorCameraUpdates(self, image):
        self.color_label.setPixmap(QPixmap(self.realsenseFrameToQImage(image)))
        # center
        self.color_label.setAlignment(Qt.AlignCenter)

    @Slot(np.ndarray)
    def StartDepthCameraUpdates(self, image):
        self.depth_label.setPixmap(
            QPixmap.fromImage(self.realsenseFrameToQImage(image))
        )
