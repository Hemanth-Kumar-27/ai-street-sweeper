"""
=========================================
Image Converter
=========================================
"""

import cv2
from PIL import Image
import customtkinter as ctk


class ImageConverter:

    @staticmethod
    def convert(frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb)

        return ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(image.width, image.height)
        )