from eval import eval_model
from checkpoint import load_from_checkpoint
import json
import logging as log
from numpy.typing import NDArray
import numpy as np


def main():
    N = 25_000

    all_rews = []

    uuid = "3eccbcb1-894f-4721-810b-fd5d0279cb73"
    model, offset = load_from_checkpoint(uuid)

    def dump_rewards(iters: int, rewards: NDArray[np.float64]):
        with open(f"models/{uuid}_rewards.jsonl", "a") as f:
            obj = {"iters": iters, "rewards": rewards.tolist()}
            json.dump(obj, f)
            f.write("\n")

    for i in range(100):
        model.learn(total_timesteps=N)
        rewards = eval_model(model)

        iters = (offset + N * (i + 1)) / 1000
        log.info(
            f"after {iters}k learning steps, mean rewards: {rewards.mean():0.2f}, std={rewards.std():0.2f}"
        )

        model.save(f"models/{uuid}_{iters}k.model")

        all_rews.append(rewards)
        dump_rewards(int(iters), rewards)


if __name__ == "__main__":
    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("## train.py ##")
    main()
