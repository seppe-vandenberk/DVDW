"""Gravity bitboards for the project's 7-column, 5-row board."""
from dataclasses import dataclass
from .game import COLS, ROWS

STRIDE = ROWS + 1
WINDOWS = tuple(sum(1 << ((c + i * dc) * STRIDE + r + i * dr) for i in range(4))
                for c in range(COLS) for r in range(ROWS)
                for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1))
                if 0 <= c + 3 * dc < COLS and 0 <= r + 3 * dr < ROWS)


def connected(bits):
    for shift in (1, STRIDE, STRIDE - 1, STRIDE + 1):
        pairs = bits & (bits >> shift)
        if pairs & (pairs >> (2 * shift)):
            return True
    return False


@dataclass(frozen=True)
class Position:
    """`own` belongs to the player about to move; rows grow from the bottom."""
    own: int = 0
    other: int = 0
    heights: tuple = (0,) * COLS

    @classmethod
    def from_board(cls, board, player):
        own = other = 0
        heights = []
        for c, column in enumerate(board):
            heights.append(sum(v != 0 for v in column))
            for r, value in enumerate(reversed(column)):
                if value == player:
                    own |= 1 << (c * STRIDE + r)
                elif value == -player:
                    other |= 1 << (c * STRIDE + r)
        return cls(own, other, tuple(heights))

    def legal(self):
        return [c for c, h in enumerate(self.heights) if h < ROWS]

    def play(self, column):
        if not 0 <= column < COLS or self.heights[column] >= ROWS:
            raise ValueError("Illegal column")
        heights = list(self.heights)
        bit = 1 << (column * STRIDE + heights[column])
        heights[column] += 1
        return Position(self.other, self.own | bit, tuple(heights))

    def terminal(self):
        if connected(self.other):
            return -1.0
        if connected(self.own):
            return 1.0
        return 0.0 if not self.legal() else None

    def board(self):
        return [[1 if self.own & (1 << (c * STRIDE + r)) else
                 -1 if self.other & (1 << (c * STRIDE + r)) else 0
                 for r in reversed(range(ROWS))] for c in range(COLS)]


def reference_move(position, depth=4):
    """Fixed alpha-beta benchmark with handcrafted open-line scores, not a solver."""
    def evaluate(p, remaining, alpha, beta):
        terminal = p.terminal()
        if terminal is not None:
            return terminal * (10000 + remaining)
        if remaining == 0:
            score = 0
            for mask in WINDOWS:
                own, other = (p.own & mask).bit_count(), (p.other & mask).bit_count()
                if not other:
                    score += (0, 1, 5, 25, 0)[own]
                if not own:
                    score -= (0, 1, 5, 25, 0)[other]
            return score
        for column in sorted(p.legal(), key=lambda c: abs(c - 3)):
            score = -evaluate(p.play(column), remaining - 1, -beta, -alpha)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha
    if depth < 1 or position.terminal() is not None:
        raise ValueError("Reference search needs a live board and positive depth")
    return max(sorted(position.legal(), key=lambda c: abs(c - 3)),
               key=lambda c: -evaluate(position.play(c), depth - 1, -float("inf"), float("inf")))
