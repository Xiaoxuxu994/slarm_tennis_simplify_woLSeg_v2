"""Registration checks for the real L515 test set, without CUDA or the data."""

import ast
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/exp0924_001_render_real_l515_0923_test.yml"
SOURCE = ROOT / "configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml"
SCRIPT = ROOT / "run_sh/render_real_l515.sh"

# Keys that say where the data and weights are. Everything else describes the
# model or its losses and must match the checkpoint's own config, or the
# weights load into a different network.
DATA_KEYS = {"dataset", "data_root", "load_from", "train_annotation",
             "eval_annotation", "exp_name"}


def _registered(name):
    tree = ast.parse((ROOT / "src/dataset/constants.py").read_text())
    return [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == name]


class RealL515ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(CONFIG.read_text())
        self.source = yaml.safe_load(SOURCE.read_text())
        self.script = SCRIPT.read_text()

    def test_model_matches_checkpoint_config(self):
        self.assertEqual(set(self.config) - DATA_KEYS, set(self.source) - DATA_KEYS)
        for key in set(self.source) - DATA_KEYS:
            self.assertEqual(self.config[key], self.source[key], key)

    def test_registration(self):
        name = self.config["dataset"][0]
        self.assertTrue(name.startswith("ball_catch"))
        found = _registered(name)
        self.assertEqual(len(found), 2)
        metadata = ast.literal_eval(next(
            value for value in found
            if any(isinstance(key, ast.Constant) and key.value == "size" for key in value.keys)))
        self.assertEqual(metadata["size"], self.config["input_size"])
        self.assertEqual(metadata["annotation_txt_file_train"], self.config["train_annotation"])
        self.assertEqual(metadata["annotation_txt_file_val"], self.config["eval_annotation"])
        self.assertEqual(len(metadata["camera_list"][3]), self.config["num_max_cameras"])

    def test_script_agrees_with_config(self):
        def var(name):
            return re.search(rf'^{name}="([^"]+)"', self.script, re.M).group(1)
        self.assertEqual(var("CONFIG"), str(CONFIG.relative_to(ROOT)))
        self.assertEqual(var("DATA_ROOT"), self.config["data_root"])
        self.assertEqual(var("MANIFEST"), self.config["eval_annotation"])
        self.assertEqual(var("CKPT"), self.config["load_from"])
        name = self.config["dataset"][0]
        listed = re.findall(r"^annotations/\S+\.json$", self.script, re.M)
        self.assertEqual(len(listed), len(var("SCENE_IDS").split(",")))
        for path in listed:
            self.assertTrue(path.startswith(f"annotations/{name}/"), path)


if __name__ == "__main__":
    unittest.main()
