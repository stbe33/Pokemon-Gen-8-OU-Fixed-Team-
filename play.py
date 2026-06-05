"""
Play against the trained bot.

1. Start the Showdown server:
       cd pokemon-showdown && node pokemon-showdown start --no-security
2. Run this script:
       python play.py
3. Open http://localhost:8000 in your browser, log in, and challenge 'RL_Bot'.
"""

import asyncio
import glob
import os
import torch

from poke_env import AccountConfiguration, LocalhostServerConfiguration

from rl_agent import DQN, OBS_SIZE
from train import FrozenPlayer, TEAM, BATTLE_FORMAT, CHECKPOINT_DIR


def load_latest_checkpoint(device):
    paths = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_v*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}/")
    latest = max(paths, key=lambda p: int(p.rsplit("_v", 1)[1].split(".")[0]))
    print(f"Loading {latest}")
    state_dict = torch.load(latest, map_location=device)
    n_actions = state_dict["net.4.weight"].shape[0]
    net = DQN(OBS_SIZE, n_actions).to(device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


async def main():
    device = torch.device("cpu")
    net = load_latest_checkpoint(device)

    bot = FrozenPlayer(
        net=net,
        device=device,
        account_configuration=AccountConfiguration("RL_Bot", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
    )

    print("Bot ready — challenge 'RL_Bot' on http://localhost:8000")
    print("Press Ctrl+C to stop.\n")
    await bot.accept_challenges(None, n_challenges=5)


if __name__ == "__main__":
    asyncio.run(main())
