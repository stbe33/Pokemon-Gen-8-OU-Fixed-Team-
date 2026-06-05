from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
import asyncio

async def main():
    player = RandomPlayer(
        account_configuration=AccountConfiguration("zmat_is_black", None),
        server_configuration=LocalhostServerConfiguration
    )
    opponent = RandomPlayer(
        account_configuration=AccountConfiguration("bot_2", None),
        server_configuration=LocalhostServerConfiguration
    )
    await player.battle_against(opponent, n_battles=1)
    
    print(f"iddy_g_is_the_goat win rate: {player.win_rate:.2f}")
    print(f"Battles completed: {player.n_finished_battles}")

asyncio.run(main())