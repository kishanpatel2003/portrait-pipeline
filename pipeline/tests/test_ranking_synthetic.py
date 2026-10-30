"""Pure ranking checks on invented metadata; no photos or model loading."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "scripts/mine_golden.py"
spec = importlib.util.spec_from_file_location("mine_golden", path)
miner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(miner)

class RankingTests(unittest.TestCase):
    def test_gate_rejects_weak_identity_and_tiny_faces(self):
        row = dict(subject=True, identity_sim=0.85, face_px=320, sharpness=80,
                   yaw=5, pitch=0, det_score=0.95, face_frac=0.10, light_mean=110)
        self.assertTrue(miner.gate(row))
        self.assertFalse(miner.gate(dict(row, identity_sim=0.20)))
        self.assertFalse(miner.gate(dict(row, face_px=50)))

    def test_close_frontal_identity_match_ranks_above_weak_profile(self):
        good = dict(identity_sim=0.85, face_px=350, sharpness=80, yaw=5,
                    pitch=0, face_frac=0.10, light_mean=110)
        weak = dict(good, identity_sim=0.51, yaw=45, face_px=185)
        self.assertGreater(miner.fitness(good), miner.fitness(weak))

    def test_perceptual_hash_distance(self):
        self.assertEqual(miner.hamming("00", "00"), 0)
        self.assertEqual(miner.hamming("00", "ff"), 8)

if __name__ == "__main__":
    unittest.main()
