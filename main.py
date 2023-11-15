import multiprocessing as mp
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from PySide6.QtWidgets import QApplication

from main_window import MainWindow


# frame collection method -> place into q
def collect_frame(color_q, depth_q, lock, event=None):
    config = rs.config()
    pipeline = rs.pipeline()
    frames = rs.frame()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # filters to clean up depth
    filters = {
        # "decimation": rs.decimation_filter(),
        "depth_to_disparity": rs.disparity_transform(True),
        "spatial": rs.spatial_filter(),
        # "temporal": rs.temporal_filter(),
        "disparity_to_depth": rs.disparity_transform(False),
        # "hole_filling": rs.hole_filling_filter(),
    }

    # fine-tunning parameters for spatial filter
    filters["spatial"].set_option(rs.option.filter_magnitude, 5)
    filters["spatial"].set_option(rs.option.filter_smooth_alpha, 1)
    filters["spatial"].set_option(rs.option.filter_smooth_delta, 50)

    i = 0

    colorizer = rs.colorizer(2)

    while not event.is_set():
        lock.acquire()
        try:
            frames = pipeline.wait_for_frames()
            color_frame = np.asanyarray(frames.get_color_frame().get_data())

            depth_frame = frames.get_depth_frame()
            # apply filters
            for f in filters.values():
                depth_frame = f.process(depth_frame)
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            depth_colormap = depth_image

            color_q.put(color_frame)
            depth_q.put(depth_colormap)

        finally:
            lock.release()
            i = i + 1

    pipeline.stop()


if __name__ == "__main__":
    # Starting the process method
    processes = []
    event = mp.Event()

    lock = mp.Lock()
    ctx = mp.get_context("spawn")
    color_q = ctx.Queue(maxsize=4)
    depth_q = ctx.Queue(maxsize=4)
    p = mp.Process(
        target=collect_frame,
        args=(
            color_q,
            depth_q,
            lock,
        ),
        kwargs={"event": event},
    )
    p.start()

    processes.append((p, event))

    # Starting Qt Application Window
    app = QApplication(sys.argv)

    window = MainWindow(color_q, depth_q, app)

    window.show()

    app.exec()

    for _, event in processes:
        event.set()

    # Now actually wait for them to shut down
    for p, _ in processes:
        print("Realsense Streams Closing...")
        p.join()
        print("Realsense Streams Closed")

    print("All processes cleaned up. Exited successfully.")
