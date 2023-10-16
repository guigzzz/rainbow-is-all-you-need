import numpy as np
from numpy.typing import NDArray

from typing import List
from x2 import X2Env
from stable_baselines3 import PPO


def eval_model(model: PPO, games: int = 10_000) -> NDArray[np.float64]:
    def make_env(seed: int) -> X2Env:
        env = X2Env()
        env.reset(seed=seed)
        return env

    envs = [make_env(s) for s in range(games)]

    def step_env_batch(model: PPO, batch: List[int]):
        all_obs = np.vstack([envs[i].get_observation() for i in batch])

        actions, _ = model.predict(all_obs, deterministic=True)

        for action, i in zip(actions, batch):
            envs[i].step(action)

    def step_envs(model: PPO, batch_size: int = 128) -> int:
        games_still_going = False

        batch = []

        for i, env in enumerate(envs):
            if not env.is_game_over():
                batch.append(i)
                games_still_going += 1

            if (len(batch) > 0) and (len(batch) % batch_size == 0):
                step_env_batch(model, batch)
                batch = []

        # last batch to cleanup
        if len(batch) > 0:
            step_env_batch(model, batch)

        return games_still_going

    while step_envs(model) > 0:
        pass

    return np.array([e.get_total_reward() for e in envs])
