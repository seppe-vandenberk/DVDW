"""Repeatable strength checks against the simulation's fixed rule-based opponent."""
import random
import time
from functools import lru_cache

from .game import drop, empty_board, outcome
from .rules import choose_rule_move
from .dqn import GAMMA, outcome_reward

EVALUATION_GAMES = 50
EVALUATION_INTERVAL = 250
BENCHMARK = "rules-v3-50"


@lru_cache(maxsize=1)
def fixed_tactics():
    from .train import tactical_suite
    return tactical_suite(per_kind=128, seed=937451)


def opponent_move(board, player, rng):
    return choose_rule_move(board, player, rng)[0]


def evaluate(agent):
    wins = losses = draws = 0
    winning_discs = []
    positions = {"first": {"wins": 0, "draws": 0, "losses": 0},
                 "second": {"wins": 0, "draws": 0, "losses": 0}}
    returns = []
    for game in range(EVALUATION_GAMES):
        # Same seeds and equal games as each color make checkpoints comparable.
        rng = random.Random(2026 + game)
        bot = 1 if game % 2 == 0 else -1
        board, player = empty_board(), 1
        while outcome(board) is None:
            column = (agent.choose(board, player, False)[0] if player == bot
                      else opponent_move(board, player, rng))
            board = drop(board, column, player)
            player = -player
        winner = outcome(board)
        result = "wins" if winner == bot else "draws" if winner == 0 else "losses"
        positions["first" if bot == 1 else "second"][result] += 1
        own_discs = sum(v == bot for col in board for v in col)
        returns.append(GAMMA ** max(0, own_discs - 1) * outcome_reward(board, bot, True))
        if winner == bot:
            wins += 1
            winning_discs.append(sum(v == bot for col in board for v in col))
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    from .train import tactical_accuracy
    tactics = tactical_accuracy(agent, fixed_tactics())
    return {"benchmark": BENCHMARK, "updates": agent.updates, "observations": agent.observations, "timestamp": time.time(),
            "positions": positions, "tactics": tactics, "mean_return": sum(returns) / len(returns),
            "games": EVALUATION_GAMES, "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / EVALUATION_GAMES,
            "score": (wins + 0.5 * draws) / EVALUATION_GAMES,
            "winning_discs": sum(winning_discs) / wins if wins else None}
