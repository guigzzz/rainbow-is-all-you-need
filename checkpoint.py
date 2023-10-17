from pathlib import Path
from typing import Tuple
from x2 import X2Env
from stable_baselines3 import PPO
import logging as log
import polars as pl
import json

MODELS = Path("models/")


def load_from_checkpoint(uuid: str) -> Tuple[PPO, int]:
    files = sorted(
        [f for f in MODELS.iterdir() if uuid in f.name and ".model" in f.name]
    )

    if len(files) == 0:
        raise Exception(f"Failed to find any models matching {uuid} in {MODELS}")

    def parse_iters(f: Path) -> int:
        return int(
            float(f.name.split("_")[1].replace("k", "").replace(".model", "")) * 1000
        )

    last, iters = next(
        reversed(sorted([(f, parse_iters(f)) for f in files], key=lambda tup: tup[1]))
    )

    log.info(f"Loading latest model: {last}")

    env = X2Env()
    model = PPO("MlpPolicy", env, seed=0, learning_rate=0.0001, batch_size=256)

    model.set_parameters(str(last))

    return model, iters


def load_rewards(uuid: str) -> pl.DataFrame:
    f = MODELS / f"{uuid}_rewards.jsonl"

    with open(f, "r") as f:
        lines = [json.loads(l.strip()) for l in f.readlines()]

    dicts = {"iter": [], "game": [], "reward": []}

    for l in lines:
        iter = l["iters"]
        for i, r in enumerate(l["rewards"]):
            dicts["iter"].append(iter)
            dicts["game"].append(i)
            dicts["reward"].append(r)

    return pl.DataFrame._from_dict(dicts)
