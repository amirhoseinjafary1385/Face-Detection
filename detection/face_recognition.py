import os 
import pickle 
import cv2 
import numpy as np

from .face_detector import FaceDetector



class FaceRecognizer:

    def __init__(self, store_path: str = "database/face_store.pkl", threhold:float = 0.56):

        self.store_path = store_path
        self.threhold = threhold
        self.detector = FaceDetector()






    def __load_store(self):
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path,'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_store(self):

        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)

        with open(self.store_path, "wb") as f:
            pickle.dump(self.known, f)

            

        