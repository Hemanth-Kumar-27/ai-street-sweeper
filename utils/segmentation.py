"""
=========================================
SegFormer Road Segmentation
=========================================
"""

import cv2
import numpy as np
import torch

from transformers import (
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
)

import config


class Segmentation:

    def __init__(self):

        print("Loading SegFormer...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = SegformerImageProcessor.from_pretrained(
            config.SEGFORMER_MODEL
        )

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            config.SEGFORMER_MODEL
        ).to(self.device)

        self.model.eval()

        self.road_class_id = self.model.config.label2id.get(
            "road",
            config.ROAD_CLASS_ID
        )

        print("SegFormer Loaded Successfully")
        print("Road class id:", self.road_class_id)

    # ------------------------------------------

    def preprocess(self, image):

        return self.processor(
            images=image,
            return_tensors="pt"
        )

    # ------------------------------------------

    def predict_mask(self, roi):

        inputs = self.preprocess(roi)
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=roi.shape[:2],
            mode="bilinear",
            align_corners=False
        )

        prediction = logits.argmax(dim=1)[0]

        return prediction.cpu().numpy()

    # ------------------------------------------

    def create_road_mask(self, prediction):

        road_mask = np.where(
            prediction == self.road_class_id,
            255,
            0
        ).astype(np.uint8)

        return road_mask

    # ------------------------------------------

    def extract_road(self, roi, road_mask):

        road_roi = cv2.bitwise_and(
            roi,
            roi,
            mask=road_mask
        )

        return road_roi

    # ------------------------------------------

    def calculate_coverage(self, road_mask):

        total_pixels = road_mask.size

        road_pixels = cv2.countNonZero(road_mask)

        coverage = (road_pixels / total_pixels) * 100

        return round(coverage, 2)

    # ------------------------------------------

    def process(self, frame, roi_mask=None):
        if roi_mask is None:
            roi_mask = np.ones(
                frame.shape[:2],
                dtype=np.uint8
            ) * 255

        target_size = (
            config.SEGMENTATION_WIDTH,
            config.SEGMENTATION_HEIGHT,
        )

        resized_frame = cv2.resize(
            frame,
            target_size,
            interpolation=cv2.INTER_AREA,
        )

        resized_roi_mask = cv2.resize(
            roi_mask,
            target_size,
            interpolation=cv2.INTER_NEAREST,
        )

        prediction = self.predict_mask(resized_frame)

        road_mask = self.create_road_mask(
            prediction
        )

        road_mask = cv2.bitwise_and(
            road_mask,
            resized_roi_mask
        )

        coverage = self.calculate_coverage(
            road_mask
        )

        road_mask_full = cv2.resize(
            road_mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        road_roi = cv2.bitwise_and(
            frame,
            frame,
            mask=road_mask_full
        )

        return (
            road_roi,
            road_mask_full,
            coverage
        )