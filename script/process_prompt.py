"""Load a prompt once and render observations without filesystem access."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from settings import SceneConfig
from frame_drawing import format_depth_info

PROMPT_PATH = Path(__file__).resolve().parent / 'prompt' / 'prompt.txt'
PROMPT_SEPARATOR = '\n--- EASG OBSERVATION TEMPLATE ---\n'


def split_prompt_template(text):
    """Optional fixed system policy followed by the per-observation template."""
    if PROMPT_SEPARATOR not in text:
        return None, text
    if text.count(PROMPT_SEPARATOR) != 1:
        raise ValueError('Prompt must contain exactly one observation separator')
    policy, _, observation = text.partition(PROMPT_SEPARATOR)
    if not policy.strip() or not observation.strip():
        raise ValueError('Prompt policy and observation template must both be nonempty')
    return policy, observation


@dataclass(frozen=True)
class PromptTemplate:
    source: str
    system: str | None
    observation: str
    config: SceneConfig

    def render(self, info, instruction):
        config = self.config
        depth_info = format_depth_info(info)
        gripper = float(np.clip(info['gripper_width_m'] / config.gripper_max_width_m, 0., 1.))
        values = {'[TCP_Z_DISTANCE_INFO]': depth_info, '[Gripper state]': f'{gripper:+.2f}', '[Task]': instruction}
        prompt = self.observation
        for key, value in values.items():
            prompt = prompt.replace(key, value)
        return prompt


def load_prompt(path, config, *, system_path=None):
    source = Path(path).read_text()
    system, observation = split_prompt_template(source)
    for token in ('[Task]', '[Gripper state]', '[TCP_Z_DISTANCE_INFO]'):
        if token not in observation:
            raise ValueError(f'Paper prompt is missing placeholder {token}')
    if system_path is not None:
        system = Path(system_path).read_text()
    return PromptTemplate(source, system, observation, config)
