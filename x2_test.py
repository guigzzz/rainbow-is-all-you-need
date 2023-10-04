from x2 import make_state, place, state_to_obs, print_grid
import numpy as np

import unittest
from typing import List

zeros = [0, 0, 0, 0, 0]


class TestX2(unittest.TestCase):
    def test_basic(self):
        state = make_state(seed=0)
        place(state, 0, 6)

        self.assertEqual(state.grid.tolist(), [[6, 0, 0, 0, 0]] + [zeros] * 4)

        place(state, 0, 6)

        self.assertEqual(state.grid.tolist(), [[7, 0, 0, 0, 0]] + [zeros] * 4)

        place(state, 0, 6)

        self.assertEqual(
            state.grid.tolist(), [[7, 0, 0, 0, 0], [6, 0, 0, 0, 0]] + [zeros] * 3
        )

        place(state, 1, 6)

        self.assertEqual(
            state.grid.tolist(), [[7, 6, 0, 0, 0], [6, 0, 0, 0, 0]] + [zeros] * 3
        )

        place(state, 1, 6)

        self.assertEqual(state.grid.tolist(), [[7, 8, 0, 0, 0]] + [zeros] * 4)

        place(state, 0, 7)

        self.assertEqual(state.grid.tolist(), [[9, 0, 0, 0, 0]] + [zeros] * 4)

    def test_compact(self):
        state = make_state(seed=0)
        place(state, 0, 6)
        place(state, 0, 7)
        place(state, 1, 6)

        self.assertEqual(state.grid.tolist(), [[8, 0, 0, 0, 0]] + [zeros] * 4)

    def test_compact2(self):
        state = make_state(seed=0)
        place(state, 0, 1)
        place(state, 0, 3)
        place(state, 0, 4)
        place(state, 0, 5)

        place(state, 1, 1)

        self.assertEqual(
            state.grid.tolist(),
            [[3, 2, 0, 0, 0], [4, 0, 0, 0, 0], [5, 0, 0, 0, 0]] + [zeros] * 2,
        )

    def test_compact3(self):
        state = make_state(seed=0)
        place(state, 0, 5)
        place(state, 1, 4)
        place(state, 2, 6)
        place(state, 0, 1)

        place(state, 1, 4)

        self.assertEqual(
            state.grid.tolist(),
            [[1, 7, 0, 0, 0]] + [zeros] * 4,
        )

    def test_compact4(self):
        grid = [
            [2.0, 7.0, 6.0, 5.0, 6.0],
            [0.0, 4.0, 5.0, 0.0, 0.0],
            [0.0, 2.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        state = make_state(seed=0)
        state.grid = np.array(grid)

        place(state, 3, 5)

        self.assertEqual(
            state.grid.tolist(),
            [
                [2.0, 7.0, 6.0, 7.0, 6.0],
                [0.0, 2.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

    def test_compact5(self):
        grid = [
            [2.0, 7.0, 6.0, 8.0, 6.0],
            [1.0, 2.0, 5.0, 4.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        state = make_state(seed=0)
        state.grid = np.array(grid)

        place(state, 2, 3)

        self.assertEqual(
            state.grid.tolist(),
            [
                [1.0, 3.0, 9.0, 4.0, 6.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )

    def test_compact6(self):
        self.run_test(
            [
                [1.0, 3.0, 9.0, 5.0, 6.0],
                [0.0, 2.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [2.0, 3.0, 9.0, 5.0, 6.0],
                [0.0, 2.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            0,
            1,
        )

    def test_compact7(self):
        self.run_test(
            [
                [4.0, 7.0, 9.0, 7.0, 6.0],
                [1.0, 4.0, 6.0, 5.0, 0.0],
                [0.0, 3.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [4.0, 7.0, 9.0, 8.0, 6.0],
                [1.0, 4.0, 6.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            3,
            5,
        )

    def test_compact8(self):
        self.run_test(
            [
                [4.0, 7.0, 9.0, 6.0, 8.0],
                [2.0, 5.0, 7.0, 0.0, 0.0],
                [0.0, 0.0, 5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [4.0, 10.0, 0.0, 6.0, 8.0],
                [2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            1,
            5,
        )

    def test_compact9(self):
        self.run_test(
            [
                [4.0, 10.0, 0.0, 6.0, 8.0],
                [2.0, 6.0, 0.0, 5.0, 0.0],
                [1.0, 5.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 3.0, 0.0],
            ],
            [
                [4.0, 10.0, 0.0, 6.0, 8.0],
                [2.0, 6.0, 0.0, 5.0, 0.0],
                [1.0, 5.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            3,
            3,
        )

    def test_compact9(self):
        self.run_test(
            [
                [8.0, 11.0, 8.0, 9.0, 7.0],
                [4.0, 0.0, 0.0, 8.0, 5.0],
                [2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [8.0, 11.0, 8.0, 9.0, 7.0],
                [4.0, 0.0, 0.0, 8.0, 5.0],
                [2.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            0,
            1,
        )

    def test_compact10(self):
        self.run_test_move(
            {
                "before": [
                    [8.0, 11.0, 8.0, 9.0, 7.0],
                    [4.0, 0.0, 0.0, 8.0, 5.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "after": [
                    [8.0, 11.0, 8.0, 9.0, 7.0],
                    [4.0, 0.0, 0.0, 8.0, 5.0],
                    [2.0, 0.0, 0.0, 6.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "next_value": 6,
                "play": 3,
            }
        )

    def test_compact11(self):
        self.run_test_move(
            {
                "before": [
                    [8.0, 11.0, 8.0, 10.0, 7.0],
                    [7.0, 6.0, 0.0, 2.0, 6.0],
                    [6.0, 0.0, 0.0, 0.0, 3.0],
                    [3.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "after": [
                    [8.0, 11.0, 8.0, 10.0, 7.0],
                    [7.0, 8.0, 0.0, 2.0, 6.0],
                    [3.0, 0.0, 0.0, 0.0, 3.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "next_value": 6,
                "play": 1,
            }
        )

    def test_compact12(self):
        self.run_test_move(
            {
                "before": [
                    [8.0, 11.0, 8.0, 10.0, 8.0],
                    [7.0, 8.0, 5.0, 2.0, 0.0],
                    [3.0, 5.0, 0.0, 5.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "after": [
                    [8.0, 0.0, 12.0, 2.0, 8.0],
                    [7.0, 0.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "next_value": 5,
                "play": 2,
            }
        )

    def test_compact13(self):
        self.run_test_move(
            {
                "before": [
                    [8.0, 7.0, 12.0, 6.0, 8.0],
                    [7.0, 6.0, 0.0, 5.0, 7.0],
                    [3.0, 0.0, 0.0, 4.0, 0.0],
                    [2.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0, 0.0],
                ],
                "after": [
                    [8.0, 9.0, 12.0, 6.0, 8.0],
                    [3.0, 0.0, 0.0, 5.0, 7.0],
                    [2.0, 0.0, 0.0, 4.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 2.0, 0.0],
                ],
                "next_value": 6,
                "play": 1,
            }
        )

    def test_state_to_obs(self):
        state = make_state(seed=0)
        state.grid[0][0] = 13
        state.grid[0][1] = 2
        state.max = 7
        state.min = 1
        state.next_play = 7

        arr = np.zeros((26,))

        obs = state_to_obs(state, arr)

        self.assertEqual(obs.tolist(), [6, 12, 1] + [0] * 23 + [1] * 5)

    def run_test_move(
        self,
        move,
    ):
        state = make_state(seed=0)
        state.grid = np.array(move["before"])
        place(state, move["play"], move["next_value"])
        self.assertEqual(state.grid.tolist(), move["after"])

    def run_test(
        self,
        start: List[List[float]],
        end: List[List[float]],
        location: int,
        value: int,
    ):
        state = make_state(seed=0)
        state.grid = np.array(start)
        place(state, location, value)
        self.assertEqual(state.grid.tolist(), end)


if __name__ == "__main__":
    unittest.main()
