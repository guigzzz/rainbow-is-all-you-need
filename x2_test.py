from x2 import make_state, place

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


if __name__ == "__main__":
    unittest.main()
