"""Check the converter and controller agree on physical actions and gripper signs."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_libero_hdf5 import converted_arm_action, format_answer
from evaluate import parse_action_text, parsed_to_libero_action
from validate_dataset import validate_answer

class RoundtripTest(unittest.TestCase):
    def test_signed_pose_and_both_gripper_states(self):
        config = {"input_min": -1, "input_max": 1,
                  "output_min": [-.05]*3 + [-.5]*3,
                  "output_max": [.05]*3 + [.5]*3}
        for gripper in (-1, 1):
            with self.subTest(gripper=gripper):
                raw = np.array([.25, -.4, .1, -.2, .3, .4, gripper])
                text = format_answer(0, converted_arm_action(raw, config))
                self.assertEqual(len(text.split("|")), 7)
                validate_answer(text, "roundtrip")
                restored, _ = parsed_to_libero_action(parse_action_text(text), config)
                np.testing.assert_allclose(restored, raw, atol=0.001)
