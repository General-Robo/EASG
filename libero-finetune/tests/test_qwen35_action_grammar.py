from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import ActionTokenGrammar, parse_action_text


class CharacterTokenizer:
    def __init__(self) -> None:
        characters = "0123456789:.,|"
        self.token_ids = {character: index for index, character in enumerate(characters)}

    def __call__(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return {"input_ids": [self.token_ids[character] for character in text]}


class ActionTokenGrammarTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = ActionTokenGrammar(CharacterTokenizer(), eos_token_ids=[100, 101])

    def encode(self, text: str) -> list[int]:
        return [self.grammar.token_id_by_character[character] for character in text]

    def assert_complete_action_allowed(self, action: str) -> None:
        prefix: list[int] = []
        for character in action:
            token_id = self.grammar.token_id_by_character[character]
            self.assertIn(token_id, self.grammar.allowed_next_token_ids(prefix))
            prefix.append(token_id)
        self.assertEqual(self.grammar.allowed_next_token_ids(prefix), [100, 101])

    def test_complete_dual_arm_action_is_allowed(self) -> None:
        action = (
            "1|1:10.74,0:2.30,1:0.05|0:12.34,1:5.67,0:0.01|1:0.25|"
            "0:9.99,1:10.00,0:0.42|1:3.14,0:0.00,1:8.50|0:1.00"
        )
        self.assert_complete_action_allowed(action)
        parsed = parse_action_text(action)
        self.assertEqual(parsed.left_translation_cm, [-10.74, 2.30, -0.05])
        self.assertEqual(parsed.right_rotation_deg, [-3.14, 0.0, -8.50])

    def test_integer_part_allows_one_or_two_digits(self) -> None:
        one_digit_prefix = self.encode("0|0:7")
        allowed = self.grammar.allowed_next_token_ids(one_digit_prefix)
        self.assertIn(self.grammar.token_id_by_character["."], allowed)
        self.assertIn(self.grammar.token_id_by_character["4"], allowed)

        two_digit_prefix = self.encode("0|0:74")
        self.assertEqual(
            self.grammar.allowed_next_token_ids(two_digit_prefix),
            [self.grammar.token_id_by_character["."]],
        )

    def test_both_arms_remain_generative(self) -> None:
        action = (
            "0|0:1.23,1:2.34,0:3.45|1:4.56,0:5.67,1:6.78|0:0.50|"
            "1:7.89,0:8.90,1:9.01|0:1.12,1:2.23,0:3.34|1:1.00"
        )
        self.assert_complete_action_allowed(action)

    def test_four_field_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 7 fields"):
            parse_action_text("0|0:0.00,0:0.00,0:0.00|0:0.00,0:0.00,0:0.00|0:1.00")

    def test_invalid_character_token_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the constrained action alphabet"):
            self.grammar.allowed_next_token_ids([999])

    def test_wrong_separator_is_not_allowed(self) -> None:
        prefix = self.encode("0")
        self.assertEqual(
            self.grammar.allowed_next_token_ids(prefix),
            [self.grammar.token_id_by_character["|"]],
        )


if __name__ == "__main__":
    unittest.main()
