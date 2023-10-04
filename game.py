from x2 import make_state, place, print_grid, State
from sshkeyboard import listen_keyboard
from random import Random


from typing import Optional
from numpy.typing import NDArray
import numpy as np
from os import path


def intTryParse(value) -> Optional[int]:
    try:
        return int(value)
    except ValueError:
        return None


import json
from io import TextIOWrapper


class Game:
    def __init__(self, file: TextIOWrapper, state: Optional[State] = None):
        self.move_file = file
        self.state = state or make_state(seed=0)

        self.print_state()

    def step_game(self, key: str):
        state = self.state
        if (play := intTryParse(key)) is None:
            return

        if play < 1 or play > 5:
            return

        before = state.grid.copy()

        result = place(state, play - 1, state.next_play)

        after = result.state.grid.copy()

        self.write_line(before, after, state.next_play, play - 1)

        if result.valid_move:
            state.next_play = state.random.randint(state.min, state.max)

        if result.game_over:
            raise Exception("Game over!")

        self.print_state()

    def write_line(
        self,
        before: NDArray[np.float64],
        after: NDArray[np.float64],
        next_value: int,
        play: int,
    ):
        obj = {
            "before": before.tolist(),
            "after": after.tolist(),
            "next_value": next_value,
            "play": play,
        }

        j = json.dumps(obj)
        self.move_file.write(j)
        self.move_file.write("\n")

    def print_state(self):
        state = self.state
        print(chr(27) + "[2J")
        print_grid(state.grid)

        print(state.next_play)


MOVES_FILE = "moves.jsonl"
if path.exists(MOVES_FILE):
    with open(MOVES_FILE, "r") as f:
        lines = f.readlines()

        start = lines[-1]


def read_existing() -> Optional[State]:
    if not path.exists(MOVES_FILE):
        return None

    with open(MOVES_FILE, "r") as f:
        lines = f.readlines()

    if len(lines) == 0:
        return None

    start = json.loads(lines[-1])
    print(start)
    return State(np.array(start["after"]), 1, 6, Random(0), 0, int(start["next_value"]))


state = read_existing()

with open("moves.jsonl", "a") as f:
    g = Game(f, state=state)

    listen_keyboard(
        on_press=g.step_game,
    )
