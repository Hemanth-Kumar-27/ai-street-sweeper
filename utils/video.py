"""
=========================================
Video Processing
=========================================
"""

import cv2
import numpy as np
import config


class VideoProcessor:

    def __init__(self):
        print("Video Path:", config.VIDEO_PATH)

        self.cap = cv2.VideoCapture(config.VIDEO_PATH)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.roi_mask = None

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

        if self.roi_mask is None:
            self.roi_mask = self._create_roi_mask(frame.shape[:2])

        return frame

    # -------------------------------------

    def get_roi(self, frame):
        if self.roi_mask is None:
            self.roi_mask = self._create_roi_mask(frame.shape[:2])

        roi = cv2.bitwise_and(
            frame,
            frame,
            mask=self.roi_mask
        )

        return roi, self.roi_mask

    def _create_roi_mask(self, size):
        h, w = size

        points = np.array([
            (int(0.35 * w), int(0.25 * h)),  # Top-left
            (int(0.65 * w), int(0.25 * h)),  # Top-right
            (int(1 * w), int(1 * h)),  # Bottom-right
            (int(0 * w), int(1 * h))  # Bottom-left
        ], dtype=np.int32)

        mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        cv2.fillPoly(
            mask,
            [points],
            255
        )

        return mask

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