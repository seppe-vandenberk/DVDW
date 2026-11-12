"""A gravity-aware Connect Four opponent with no learned weights."""
import random

from .game import COLS, ROWS, drop, legal_moves, outcome

RULE_MODEL = "rules"
WINDOWS = [tuple((c + i * dc, r + i * dr) for i in range(4))
           for c in range(COLS) for r in range(ROWS)
           for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1))
           if 0 <= c + 3 * dc < COLS and 0 <= r + 3 * dr < ROWS]


def sequence_value(board, player):
    """Longest contiguous run and growth potential in lines that can still win."""
    longest, potential = 0, 0
    for window in WINDOWS:
        values = [board[c][r] for c, r in window]
        if -player in values:
            continue
        run = 0
        for value in values:
            run = run + 1 if value == player else 0
            longest = max(longest, run)
        potential += (0, 1, 6, 30, 10000)[values.count(player)]
    return longest, potential


def winning_moves(board, player):
    return [column for column in legal_moves(board)
            if outcome(drop(board, column, player)) == player]


def choose_rule_move(board, player, rng=None):
    """Return a random opening, then follow tactical rules with random ties."""
    rng = rng if rng is not None else random
    legal = legal_moves(board)
    if outcome(board) is not None:
        raise ValueError("Cannot play on a finished board")
    if not any(value == player for column in board for value in column):
        return rng.choice(legal), "Random opening move"
    wins = winning_moves(board, player)
    if wins:
        return rng.choice(wins), "Complete four in a row"

    threats = winning_moves(board, -player)
    candidates = {column: drop(board, column, player) for column in legal}
    replies = {column: len(winning_moves(after, -player))
               for column, after in candidates.items()}
    # Prefer safety before building a sequence; if trapped, minimize winning replies.
    safest = min(replies.values())
    choices = [column for column in legal if replies[column] == safest]
    own_length = sequence_value(board, player)[0]
    opposing_length = sequence_value(board, -player)[0]
    defend_pair = opposing_length == 2 and own_length <= 1

    def rank(column):
        after = candidates[column]
        own = sequence_value(after, player)
        other = sequence_value(after, -player)
        center = -abs(column - (COLS - 1) / 2)
        if threats or defend_pair:
            return (-other[0], -other[1], own[0], own[1], center)
        return (own[0], own[1], -other[1], center)

    ranks = {column: rank(column) for column in choices}
    best = max(ranks.values())
    column = rng.choice([column for column in choices if ranks[column] == best])
    reason = ("Block an immediate win" if threats else
              "Block an opposing pair while holding only singles" if defend_pair else
              "Extend an own sequence and favor the center")
    return column, reason
