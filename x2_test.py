from x2 import make_state, place, state_to_obs, print_grid
import numpy as np

import unittest

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

    # def test_state_to_obs(self):
    #     state = make_state(seed=0)
    #     state.grid[0][0] = 13
    #     state.grid[0][1] = 2
    #     state.max = 7
    #     state.min = 1
    #     state.next_play = 7

    #     arr = np.zeros((26,))

    #     obs = state_to_obs(state, arr)

    #     self.assertEqual(obs.tolist(), [6, 12, 1] + [0] * 23)


if __name__ == "__main__":
    unittest.main()
