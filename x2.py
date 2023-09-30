N = 5

INCREMENT_FACTOR = 2

from typing import List, Optional, Tuple
from dataclasses import dataclass
from random import Random


@dataclass
class State:
    grid: List[List[int]]
    min: int
    max: int
    random: Random
    num_invalid_moves: int
    next_play: int


def make_state(seed: Optional[int] = None) -> State:
    grid = [[0 for _ in range(N)] for _ in range(N)]
    min = 1
    max = 6
    random = Random(seed)
    next_play = random.randint(min, max)
    return State(grid, min, max, random, 0, next_play)


def try_combine(grid: List[List[int]], i: int, j: int) -> Optional[List[List[int]]]:
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


def move_down_once(grid: List[List[int]]) -> Optional[Tuple[int, int]]:
    for i in range(N - 1):
        for j in range(N):
            if grid[i][j] == 0 and grid[i + 1][j] > 0:
                grid[i][j] = grid[i + 1][j]
                grid[i + 1][j] = 0

                return (i, j)

    return None


def is_game_over(grid: List[List[int]]) -> bool:
    for row in grid:
        for col in row:
            if col == 0:
                return False

    return True


def check_update_min_max(state: State) -> Optional[State]:
    grid = state.grid
    mx = max([max(r) for r in grid])

    if mx >= state.max * INCREMENT_FACTOR:
        state.min += 1
        state.max += 1


@dataclass
class PlaceResult:
    valid_move: bool
    game_over: bool
    state: State


def solve(grid: List[List[int]], start_i: int, start_j: int) -> List[List[int]]:
    filled_i = start_i
    filled_j = start_j

    dirty = True
    while dirty:
        dirty = False

        new_grid = try_combine(grid, filled_i, filled_j)
        if new_grid is not None:
            dirty = True
            filled_i_j = move_down_once(new_grid)

            if filled_i_j is not None:
                filled_i, filled_j = filled_i_j

        filled_i_j = move_down_once(grid)
        if filled_i_j is not None:
            dirty = True
            filled_i, filled_j = filled_i_j

    return grid


def place(state: State, location: int, value: int) -> PlaceResult:
    if value <= 0:
        raise Exception(f"Invalid value: {value}")

    grid = state.grid
    if location < 0 or location > 4:
        raise Exception(f"Invalid location: {location}. Must be 0 <= location <= 4.")

    if grid[-1][location] > 0:
        state.num_invalid_moves += 1
        if state.num_invalid_moves > 5:
            return PlaceResult(False, True, state)
        # colummn full already, do nothing
        return PlaceResult(False, False, state)

    state.num_invalid_moves = 0

    # find spot
    filled_i = None
    for i in range(N):
        if grid[i][location] == 0:
            grid[i][location] = value
            filled_i = i
            break

    if filled_i is None:
        raise Exception("impossible")

    filled_j = location

    solve(grid, filled_i, filled_j)

    update_min_max = check_update_min_max(state)
    if update_min_max is not None:
        # delete all prev mins
        prev_min = state.min - 1
        for i in range(N):
            for j in range(N):
                if state.grid[i][j] == prev_min:
                    state.grid[i][j] = 0

        filled_i_j = move_down_once(state.grid)
        if filled_i_j is not None:
            solve(state.grid, filled_i_j[0], filled_i_j[1])

    state.grid = grid

    game_over = is_game_over(grid)
    return PlaceResult(True, game_over, state)


def print_grid(grid: List[List[int]]):
    for row in grid:
        print(" ".join(["----" if c == 0 else str(c).zfill(4) for c in row]))


import gymnasium as gym
from typing import Dict, Any
import numpy as np
from numpy.typing import NDArray


def state_to_obs(state: State) -> NDArray[np.int64]:
    normalise = state.max - 6
    flat = [state.next_play] + [c for r in state.grid for c in r]
    arr = np.array(flat, dtype=np.int64) - normalise / (6 * INCREMENT_FACTOR)
    return arr


class X2Env(gym.Env[NDArray[np.int64], np.int64]):
    def __init__(self) -> None:
        super().__init__()

        self._state: Optional[State] = None

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            0,
            12,
        )

        self.__info: Dict[str, str] = {}

    def step(self, action: np.int64):
        state = self.__get_state()

        normalise = state.max - 6
        action += normalise

        result = place(state, int(action), state.next_play)

        if result.valid_move:
            state.next_play = self.__generate_tile()

        obs = state_to_obs(result.state)

        reward = 1 if result.valid_move else -1
        return (
            obs,
            reward,
            result.game_over,
            False,
            self.__info,
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        self._state = make_state(seed)
        self.action_space = gym.spaces.Discrete(5, seed)

        return state_to_obs(self._state), self.__info

    def __generate_tile(self) -> int:
        s = self.__get_state()
        return s.random.randint(s.min, s.max)

    def __get_state(self) -> State:
        if self._state is None:
            self._state = make_state()

        return self._state

    def render(self) -> None:
        return None
