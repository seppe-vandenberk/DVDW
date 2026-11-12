"""Boards are column-major: board[column][row], row zero at the top."""
COLS, ROWS = 7, 5


def empty_board():
    return [[0] * ROWS for _ in range(COLS)]


def legal_moves(board):
    return [c for c in range(COLS) if board[c][0] == 0]


def drop(board, column, player):
    if column not in legal_moves(board):
        raise ValueError("Column is full or outside 0..6")
    result = [col[:] for col in board]
    row = max(r for r in range(ROWS) if result[column][r] == 0)
    result[column][row] = player
    return result


def winners(board):
    found = set()
    for c in range(COLS):
        for r in range(ROWS):
            p = board[c][r]
            if not p:
                continue
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                if all(0 <= c + i * dc < COLS and 0 <= r + i * dr < ROWS
                       and board[c + i * dc][r + i * dr] == p for i in range(4)):
                    found.add(p)
    return found


def outcome(board):
    won = winners(board)
    if won:
        return next(iter(won))
    return None if legal_moves(board) else 0


def validate_board(board):
    if len(board) != COLS or any(len(col) != ROWS for col in board):
        raise ValueError("Expected 7 columns, each containing 5 rows")
    if any(type(v) is not int or v not in (-1, 0, 1) for col in board for v in col):
        raise ValueError("Cells must be integers: -1, 0, or 1")
    for col in board:
        occupied = False
        for v in col:
            if v:
                occupied = True
            elif occupied:
                raise ValueError("Floating disks are not allowed; row 0 is the top")
    plus = sum(v == 1 for col in board for v in col)
    minus = sum(v == -1 for col in board for v in col)
    if plus not in (minus, minus + 1):
        raise ValueError("Player 1 starts; disk counts must alternate")
    won = winners(board)
    if len(won) > 1 or (1 in won and plus != minus + 1) or (-1 in won and plus != minus):
        raise ValueError("Winner is inconsistent with disk counts")
    if won:
        # A legal terminal board must have a possible last disk that caused the win.
        player = next(iter(won))
        for c, col in enumerate(board):
            for r, value in enumerate(col):
                if value:
                    if value == player:
                        previous = [x[:] for x in board]
                        previous[c][r] = 0
                        if not winners(previous):
                            return
                    break
        raise ValueError("Board contains moves after the game ended")


def next_player(board):
    return 1 if sum(v for col in board for v in col) == 0 else -1


def encode(board, player):
    return [[v * player for v in col] for col in board]
