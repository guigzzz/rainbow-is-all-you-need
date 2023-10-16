N = 5

INCREMENT_FACTOR = 2

from typing import Optional, Tuple
from dataclasses import dataclass
from random import Random
from numpy.typing import NDArray
import numpy as np


@dataclass
class State:
    grid: NDArray[np.float64]
    min: int
    max: int
    random: Random
    num_invalid_moves: int
    next_play: int


def copy(state: State) -> State:
    return State(
        state.grid.copy(),
        state.min,
        state.max,
        Random(state.random.seed),
        state.num_invalid_moves,
        state.next_play,
    )


def make_state(seed: Optional[int] = None) -> State:
    grid = np.zeros((5, 5))
    min = 1
    max = 6
    random = Random(seed)
    next_play = random.randint(min, max)
    return State(grid, min, max, random, 0, next_play)


def try_combine(
    grid: NDArray[np.float64], i: int, j: int
) -> Optional[NDArray[np.float64]]:
    value = grid[i][j]
    if value == 0:
        raise Exception("nope")

    count = 0
    if i > 0 and grid[i - 1][j] == value:
        count += 1
        grid[i - 1][j] = 0
    if j > 0 and grid[i][j - 1] == value:
        count += 1
        grid[i][j - 1] = 0
    if j < 4 and grid[i][j + 1] == value:
        count += 1
        grid[i][j + 1] = 0

    if count > 0:
        grid[i][j] = value + count
        return grid

    return None


def move_down_once(grid: NDArray[np.float64]) -> Optional[Tuple[int, int]]:
    for i in range(N - 1):
        for j in range(N):
            if grid[i][j] == 0 and grid[i + 1][j] > 0:
                grid[i][j] = grid[i + 1][j]
                grid[i + 1][j] = 0

                return (i, j)

    return None


def try_move_down(
    grid: NDArray[np.float64], i: int, j: int
) -> Optional[Tuple[int, int]]:
    if i == 0:
        return None

    if i > 0 and grid[i][j] > 0 and grid[i - 1][j] == 0:
        grid[i - 1][j] = grid[i][j]
        grid[i][j] = 0
        return (i - 1, j)

    return None


def is_game_over(grid: NDArray[np.float64]) -> bool:
    for row in grid:
        for col in row:
            if col == 0:
                return False

    return True


def check_update_min_max(state: State) -> bool:
    mx = state.grid.max()

    if mx >= state.max + 6:
        state.min += 1
        state.max += 1
        return True

    return False


@dataclass
class PlaceResult:
    valid_move: bool
    game_over: bool
    state: State


def solve(grid: NDArray[np.float64], start_i: int, start_j: int) -> NDArray[np.float64]:
    filled_i = start_i
    filled_j = start_j

    dirty = True
    while dirty:
        dirty = False

        new_grid = try_combine(grid, filled_i, filled_j)
        while new_grid is not None:
            dirty = True

            filled_i_j = try_move_down(grid, filled_i, filled_j)
            if filled_i_j is not None:
                filled_i, filled_j = filled_i_j

            new_grid = try_combine(grid, filled_i, filled_j)

        filled_i_j = move_down_once(grid)

        if filled_i_j is not None:
            dirty = True
            filled_i, filled_j = filled_i_j

    return grid


def sanity_check_grid(
    grid: NDArray[np.float64], state: State, location: int, value: int
):
    min, max = state.min, state.max
    for i in range(N - 1):
        for j in range(N):
            v = grid[i][j]
            if v == 0 and grid[i + 1][j] > 0:
                print_grid(grid)
                print(min, max, state.next_play, location, value)
                raise Exception(f"BUG")

            if v > 0 and (v < min or v > max + 6):
                raise Exception(f"Out of minmax bounds: {v} is < {min} or > {max + 6}")


def find_prev_min(state: State) -> Optional[Tuple[int, int]]:
    prev_min = state.min - 1
    for i in range(N):
        for j in range(N):
            if state.grid[i][j] == prev_min:
                state.grid[i][j] = 0

                if i < 4 and state.grid[i + 1][j] > 0:
                    filled_i_j = try_move_down(state.grid, i + 1, j)
                    if filled_i_j is not None:
                        return filled_i_j

    return None


def place(state: State, location: int, value: int) -> PlaceResult:
    if value <= 0:
        raise Exception(f"Invalid value: {value}")

    grid = state.grid
    if location < 0 or location > 4:
        raise Exception(f"Invalid location: {location}. Must be 0 <= location <= 4.")

    if grid[-1][location] > 0 and grid[-1][location] != value:
        state.num_invalid_moves += 1
        if state.num_invalid_moves > 5:
            return PlaceResult(False, True, state)
        # colummn full already, do nothing
        # EXCEPT if value inserted matches top of the column
        return PlaceResult(False, False, state)

    state.num_invalid_moves = 0

    # find spot
    filled_i = None
    for i in range(N):
        if grid[i][location] == 0:
            grid[i][location] = value
            filled_i = i
            break

        # placing on full column. Just increment the top value
        if i == N - 1 and grid[i][location] == value:
            filled_i = i
            grid[i][location] += 1

    if filled_i is None:
        raise Exception("impossible")

    filled_j = location

    solve(grid, filled_i, filled_j)

    if check_update_min_max(state):
        # delete all prev mins
        prev_min_ij = find_prev_min(state)
        while prev_min_ij is not None:
            solve(state.grid, prev_min_ij[0], prev_min_ij[1])
            prev_min_ij = find_prev_min(state)

    sanity_check_grid(grid, state, location, value)

    game_over = is_game_over(grid)
    return PlaceResult(True, game_over, state)


def apply_move(state: State, move: int) -> Tuple[State, int, bool]:
    result = place(state, move, state.next_play)

    if result.valid_move:
        state.next_play = state.random.randint(state.min, state.max)

    reward = 1 if result.valid_move else 0

    return (result.state, reward, result.game_over)


def print_grid(grid: NDArray[np.float64]):
    for row in grid:
        print(" ".join(["----" if c == 0 else str(int(c)).zfill(4) for c in row]))


import gymnasium as gym
from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray


def state_to_obs(state: State, arr: NDArray[np.float64]) -> NDArray[np.float64]:
    offset = state.min - 1
    arr[0] = state.next_play - offset
    flat = state.grid.flatten()
    arr[1:] = np.where(flat == 0, 0, flat - offset)

    return arr


class X2Env(gym.Env[NDArray[np.float64], np.int64]):
    def __init__(self) -> None:
        super().__init__()

        self._state: Optional[State] = None

        self.action_space = gym.spaces.Discrete(5)

        self.__observation = np.zeros((26,))
        self.observation_space = gym.spaces.Box(
            0, 12, shape=self.__observation.shape, dtype=np.float64
        )

        self.__is_game_over = False
        self.__total_reward = 0

        self.__info: Dict[str, str] = {}

    def step(self, action: np.int64):
        state = self.get_state()

        result = place(state, int(action), state.next_play)

        if result.valid_move:
            state.next_play = self.__generate_tile()

        obs = state_to_obs(result.state, self.__observation)

        self.__is_game_over = result.game_over

        reward = 1 if result.valid_move else -1
        self.__total_reward += reward
        return (
            obs,
            reward,
            result.game_over,
            False,
            self.__info,
        )

    def reset(self, *, seed: Optional[int] = None, _: Optional[Dict[str, Any]] = None):
        self._state = make_state(seed)
        self.action_space = gym.spaces.Discrete(5, seed)

        return state_to_obs(self._state, self.__observation), self.__info

    def __generate_tile(self) -> int:
        s = self.get_state()
        return s.random.randint(s.min, s.max)

    def get_state(self) -> State:
        if self._state is None:
            self._state = make_state()

        return self._state

    def get_observation(self) -> NDArray[np.float64]:
        return self.__observation

    def is_game_over(self) -> bool:
        return self.__is_game_over

    def get_total_reward(self) -> int:
        return self.__total_reward

    def render(self) -> None:
        return None
