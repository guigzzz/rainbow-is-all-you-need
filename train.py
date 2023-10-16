from eval import eval_model
from checkpoint import load_from_checkpoint
import json


def main():
    N = 25_000

    all_rews = []

    uuid = "3eccbcb1-894f-4721-810b-fd5d0279cb73"
    model, offset = load_from_checkpoint(uuid)

    for i in range(1, 100):
        model.learn(total_timesteps=N)
        rewards = eval_model(model)

        iters = (offset + N * (i + 1)) / 1000
        print(
            f"after {iters}k learning steps, mean rewards: {rewards.mean():0.2f}, std={rewards.std():0.2f}"
        )

        model.save(f"models/{uuid}_{iters}k.model")

        all_rews.append(rewards)

    with open(f"models/{uuid}_rewards.json", "w") as f:
        obj = {(offset + N * (i + 1)) / 1000: rews for i, rews in enumerate(all_rews)}
        json.dump(obj, f)


if __name__ == "__main__":
    print("## train.py ##")
    main()
