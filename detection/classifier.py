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

    def predict(self, landmarks) -> tuple[str, float]:
        """Return (predicted_label, confidence) from a list of 33 landmarks."""
        row   = [v for lm in landmarks for v in (lm.x, lm.y, lm.z)]
        probs = self._clf.predict_proba([row])[0]
        idx   = int(np.argmax(probs))
        return self._enc.classes_[idx], float(probs[idx])
