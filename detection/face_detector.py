import cv2 
import os 
import requests

import numpy as np


class FaceDetector:

    def __init__(self):
        if cascade_path is None:
            cascade_path = cv2.data.hearcascade + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                raise RuntimeError(f"failed to load cascade:  {cascade_path}")


            










