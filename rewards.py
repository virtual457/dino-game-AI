from config import REWARD_DEATH


def calculate_reward(action, game_over):
    if game_over:
        return REWARD_DEATH
    return 10.0
