"""
MobileNetV2 Debris Predictor
"""

import json
import cv2
import numpy as np
import tensorflow as tf

from AI_Street_Sweeper import config


class DebrisPredictor:

    def __init__(self):

        print("Loading MobileNetV2...")

        # Load trained model
        self.model = tf.keras.models.load_model(
            config.MODEL_PATH
        )

        # Load class names
        with open(
            config.CLASS_NAMES_PATH,
            "r"
        ) as f:

            self.class_names = json.load(f)

        print(
            "Class Names:",
            self.class_names
        )

        print(
            "MobileNetV2 Loaded Successfully"
        )

    # =================================================
    # PREPROCESS IMAGE
    # =================================================

    def preprocess(self, road_roi):
        image = cv2.resize(
            road_roi,
            (config.INPUT_SIZE, config.INPUT_SIZE)
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = image.astype(
            np.float32
        )

        image = np.expand_dims(
            image,
            axis=0
        )

        return image

    # =================================================
    # PREDICT
    # =================================================

    def predict(self, road_roi):
        image = self.preprocess(road_roi)

        prediction = self.model.predict(
            image,
            verbose=0
        )[0]

        class_index = int(
            np.argmax(prediction)
        )

        confidence = round(
            float(prediction[class_index]) * 100,
            2
        )

        class_name = self.class_names[class_index]

        return (
            class_name.lower(),
            confidence
        )