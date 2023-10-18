from eval import eval_model
from checkpoint import load_from_checkpoint
import json
import logging as log
from numpy.typing import NDArray
import numpy as np
from uuid import uuid4
from x2 import X2Env
from stable_baselines3 import PPO

from multiprocessing import Process, Queue, set_start_method

set_start_method(method="spawn", force=True)


def make_env_and_model():
    env = X2Env()
    model = PPO("MlpPolicy", env, seed=0, learning_rate=0.0001, batch_size=256)
    return model


def eval_thread(queue: Queue, uuid: str):
    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info(f"[EVAL] Hello world!")

    def dump_rewards(iters: int, rewards: NDArray[np.float64]):
        with open(f"models/{uuid}_rewards.jsonl", "a") as f:
            obj = {"iters": iters, "rewards": rewards.tolist()}
            json.dump(obj, f)
            f.write("\n")

    while True:
        tup = queue.get()
        if tup is None:
            log.info(f"[EVAL] Stopping")
            return

        (model_path, iters) = tup
        log.info(f"[EVAL] Got {model_path} @ {iters}k steps")

        model = make_env_and_model()
        model.set_parameters(str(model_path))

        rewards = eval_model(model)

        log.info(
            f"[EVAL] after {iters}k learning steps, mean rewards: {rewards.mean():0.2f}, std={rewards.std():0.2f}"
        )

        dump_rewards(int(iters), rewards)


def main():
    N = 50_000

    env = X2Env()
    model = PPO("MlpPolicy", env, seed=0, learning_rate=0.0001, batch_size=256)

    # uuid = str(uuid4())
    # offset = 0

    # # Two tile lookahead
    uuid = "0fdef917-edc8-4a87-a2b0-be201cdb3d91"
    model, offset = load_from_checkpoint(uuid, model)

    # # Single tile lookahead
    # uuid = "3eccbcb1-894f-4721-810b-fd5d0279cb73"
    # model, offset = load_from_checkpoint(uuid, model)

    q = Queue()
    p = Process(target=eval_thread, args=(q, uuid))
    p.start()

    for i in range(100):
        model.learn(total_timesteps=N)

        iters = (offset + N * (i + 1)) / 1000

        path = f"models/{uuid}_{iters}k.model"
        model.save(path)

        q.put((path, iters))

        log.info(f"[TRAIN] Done {iters}k learning steps")

    q.put(None)
    p.join()


if __name__ == "__main__":
    log.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=log.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("## train.py ##")
    main()
