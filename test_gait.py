import unittest
import numpy as np
from gait_engine import extract_gait_features, cosine_similarity

class TestGaitEngine(unittest.TestCase):
    def test_extract_gait_features_too_short(self):
        # Less than 10 frames should return None
        dummy_seq = np.zeros((5, 33, 4))
        features = extract_gait_features(dummy_seq)
        self.assertIsNone(features)

    def test_extract_gait_features_valid(self):
        # Valid frame count (e.g. 30 frames of dummy keypoints)
        np.random.seed(42)
        dummy_seq = np.random.rand(30, 33, 4)
        features = extract_gait_features(dummy_seq)
        self.assertIsNotNone(features)
        # 6 signals * 4 stats (mean, std, min, max) = 24 features
        self.assertEqual(len(features), 24)

    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        
        v3 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

if __name__ == "__main__":
    unittest.main()
