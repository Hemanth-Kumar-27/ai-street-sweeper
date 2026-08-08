"""
=========================================
Video Processing
=========================================
"""

import cv2
import numpy as np
from AI_Street_Sweeper import config


class VideoProcessor:

    def __init__(self):
        print("Video Path:", config.VIDEO_PATH)

        self.cap = cv2.VideoCapture(config.VIDEO_PATH)

        print("Opened:", self.cap.isOpened())

    # -------------------------------------

    def get_frame(self):

        ret, frame = self.cap.read()

        if not ret:

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame = self.cap.read()

        frame = cv2.resize(
            frame,
            (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        )

        return frame

    # -------------------------------------

    def get_roi(self, frame):
        h, w = frame.shape[:2]

        # Larger trapezoidal ROI
        points = np.array([
            (int(0.2 * w), int(0.0 * h)),  # Top-left
            (int(0.7 * w), int(0.0 * h)),  # Top-right
            (int(0.98 * w), int(0.98 * h)),  # Bottom-right
            (int(0.02 * w), int(0.98 * h))  # Bottom-left
        ], dtype=np.int32)

        # ROI mask
        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [points],
            255
        )

        # ROI image for dashboard display
        roi = cv2.bitwise_and(
            frame,
            frame,
            mask=mask
        )

        return roi, mask

    def get_cnn_roi(self, frame):
        h, w = frame.shape[:2]

        points = np.float32([
            [0.15 * w, 0.35 * h],
            [0.85 * w, 0.35 * h],
            [0.98 * w, 0.98 * h],
            [0.02 * w, 0.98 * h]
        ])

        output_points = np.float32([
            [0, 0],
            [224, 0],
            [224, 224],
            [0, 224]
        ])

        matrix = cv2.getPerspectiveTransform(
            points,
            output_points
        )

        cnn_roi = cv2.warpPerspective(
            frame,
            matrix,
            (224, 224)
        )

        return cnn_roi
    # -------------------------------------

    def release(self):

        self.cap.release()