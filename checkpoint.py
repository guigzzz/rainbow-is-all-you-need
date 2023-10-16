from pathlib import Path
from typing import Tuple
from x2 import X2Env
from stable_baselines3 import PPO
import logging as log


def load_from_checkpoint(uuid: str) -> Tuple[PPO, int]:
    p = Path("models/")
    files = sorted([f for f in p.iterdir() if uuid in f.name])

    if len(files) == 0:
        raise Exception(f"Failed to find any models matching {uuid} in {p}")

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
