from x2 import X2Env

env = X2Env()
env.reset(seed=0)

for _ in range(100_000):
    action = env.action_space.sample()
    _, _, done, _, _ = env.step(action)
    if done:
        env.reset(seed=0)
