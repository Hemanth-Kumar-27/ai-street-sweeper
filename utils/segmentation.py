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

        self.processor = SegformerImageProcessor.from_pretrained(
            config.SEGFORMER_MODEL
        )

        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            config.SEGFORMER_MODEL
        )

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

        with torch.no_grad():
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

        # Run SegFormer on the provided frame
        prediction = self.predict_mask(frame)

        # Create road mask
        road_mask = self.create_road_mask(
            prediction
        )

        # Restrict to trapezoid
        road_mask = cv2.bitwise_and(
            road_mask,
            roi_mask
        )

        # Extract road
        road_roi = cv2.bitwise_and(
            frame,
            frame,
            mask=road_mask
        )

        # Calculate coverage
        coverage = self.calculate_coverage(
            road_mask
        )

        return (
            road_roi,
            road_mask,
            coverage
        )