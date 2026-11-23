import cv2 
import numpy as np

from .face_recognition import FaceRecognizer
from .face_detector import FaceDetector


class FaceComparator:
    def __init__(self, threshold: float = 0.60):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.threshold = threshold

    # ------------------------------------------------------------------
    def _first_face(self, image_path: str):
        faces = self.detector.detect(image_path)
        if not faces:
            return None, None
        best = max(faces, key=lambda f: f["w"] * f["h"])
        crop = self.detector.crop_face(image_path, best)
        enc = self.recognizer._encode(crop)
        return best, enc


    def compare(self, path1: str, path2: str) -> dict:

        """
        Returns:
            {
                'similarity': float,
                'distance': float,
                'match': bool,
                'faces_found': int,
                'threshold': float
            
            }
        """

        f1, e1 = self._first_face(path1)
        f2, e2 = self._first_face(path2)

        faces_found = int(f1 is not None) + int(f2 is not None)

        if e1 is None or e2 is None :
            return {
                'similirity': 0.0,
                'distance': 1.0,
                'match': False,
                'faces_found': faces_found,
                'threhold': self.threshold,
                'error':  "One or both images contain no detectable face.",
            }

        sim = float(np.dot(e1, e2))
        sim = max(0.0, min(1.0, sim))                

    