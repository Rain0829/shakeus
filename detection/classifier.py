import pickle
import numpy as np
from config import CLASSIFIER_PATH


class PoseClassifier:
    """Wraps the trained RandomForest pose classifier."""

    def __init__(self):
        with open(CLASSIFIER_PATH, "rb") as f:
            saved = pickle.load(f)
        self._clf = saved["classifier"]
        self._enc = saved["label_encoder"]
        self.labels = list(self._enc.classes_)
        print(f"Classifier loaded. Labels: {self.labels}")

    @staticmethod
    def _normalize(landmarks):
        hip_x = (landmarks[23].x + landmarks[24].x) / 2
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        scale = abs(landmarks[11].x - landmarks[12].x) + 1e-6
        row = []
        for lm in landmarks:
            row.extend([
                (lm.x - hip_x) / scale,
                (lm.y - hip_y) / scale,
                lm.z,
            ])
        return row

    def predict(self, landmarks) -> tuple[str, float]:
        """Return (predicted_label, confidence) from a list of 33 landmarks."""
        row   = self._normalize(landmarks)
        probs = self._clf.predict_proba([row])[0]
        idx   = int(np.argmax(probs))
        return self._enc.classes_[idx], float(probs[idx])
