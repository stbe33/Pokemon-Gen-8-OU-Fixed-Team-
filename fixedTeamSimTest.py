"""
Simulate Gen 8 OU mirror matches and record full battle states.

Run:
    python simulate.py

Requires local Showdown server:
    cd pokemon-showdown && node pokemon-showdown start --no-security
"""

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from poke_env.player import RandomPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

N_BATTLES     = 100
BATTLE_FORMAT = "gen8ou"
OUTPUT_FILE   = "battle_log.json"

TEAM = """
Kingler @ Life Orb
Ability: Sheer Force
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Swords Dance
- Knock Off
- Liquidation
- High Horsepower

Ferrothorn @ Leftovers
Ability: Iron Barbs
EVs: 252 HP / 224 Def / 32 SpD
Impish Nature
IVs: 0 Atk
- Stealth Rock
- Leech Seed
- Body Press
- Thunder Wave

Tornadus-Therian @ Heavy-Duty Boots
Ability: Regenerator
EVs: 248 HP / 92 Def / 168 Spe
Timid Nature
- Hurricane
- Taunt
- U-turn
- Defog

Slowking-Galar @ Assault Vest
Ability: Regenerator
EVs: 248 HP / 12 Def / 204 SpA / 44 SpD
Modest Nature
IVs: 0 Atk
- Future Sight
- Sludge Bomb
- Flamethrower
- Scald

Garchomp @ Leftovers
Ability: Rough Skin
EVs: 48 HP / 148 Atk / 60 SpD / 252 Spe
Adamant Nature
- Swords Dance
- Earthquake
- Scale Shot
- Aqua Tail

Weavile @ Choice Band
Ability: Pickpocket
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Ice Shard
- Icicle Crash
- Knock Off
- Beat Up
"""


# ── State dataclasses ─────────────────────────────────────────────────────────

@dataclass
class PokemonState:
    species:          str
    hp_fraction:      float
    fainted:          bool
    active:           bool
    status:           Optional[str]   # BRN, PAR, PSN, etc.
    boosts:           dict            # atk/def/spa/spd/spe/acc/eva

@dataclass
class TurnSnapshot:
    turn:             int
    active_pokemon:   str
    opp_active:       str
    my_team:          List[PokemonState]
    opp_team:         List[PokemonState]
    weather:          Optional[str]
    fields:           List[str]
    side_conditions:  List[str]
    available_moves:  List[str]
    available_switches: List[str]
    action_taken:     Optional[str]   # filled in after the fact if we know it

@dataclass
class BattleRecord:
    battle_id:        str
    won:              Optional[bool]
    total_turns:      int
    my_fainted:       int
    opp_fainted:      int
    turns:            List[TurnSnapshot] = field(default_factory=list)


# ── Tracking player ───────────────────────────────────────────────────────────

class TrackingPlayer(RandomPlayer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records: List[BattleRecord] = []
        self._current_records = {}   # battle_tag -> BattleRecord

    def _get_pokemon_state(self, pokemon) -> PokemonState:
        try:
            boosts = {
                "atk": pokemon.boosts.get("atk", 0),
                "def": pokemon.boosts.get("def", 0),
                "spa": pokemon.boosts.get("spa", 0),
                "spd": pokemon.boosts.get("spd", 0),
                "spe": pokemon.boosts.get("spe", 0),
                "acc": pokemon.boosts.get("accuracy", 0),
                "eva": pokemon.boosts.get("evasion", 0),
            }
        except Exception:
            boosts = {}

        return PokemonState(
            species     = pokemon.species,
            hp_fraction = pokemon.current_hp_fraction,
            fainted     = pokemon.fainted,
            active      = pokemon.active,
            status      = str(pokemon.status) if pokemon.status else None,
            boosts      = boosts,
        )

    def _snapshot_turn(self, battle) -> TurnSnapshot:
        my_team  = [self._get_pokemon_state(p) for p in battle.team.values()]
        opp_team = [self._get_pokemon_state(p) for p in battle.opponent_team.values()]

        return TurnSnapshot(
            turn              = battle.turn,
            active_pokemon    = battle.active_pokemon.species if battle.active_pokemon else "unknown",
            opp_active        = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else "unknown",
            my_team           = my_team,
            opp_team          = opp_team,
            weather           = str(battle.weather) if battle.weather else None,
            fields            = [str(f) for f in battle.fields],
            side_conditions   = [str(c) for c in battle.side_conditions],
            available_moves   = [m.id for m in battle.available_moves],
            available_switches= [p.species for p in battle.available_switches],
            action_taken      = None,
        )

    def choose_move(self, battle):
        # Record state before choosing
        tag = battle.battle_tag
        if tag not in self._current_records:
            self._current_records[tag] = BattleRecord(
                battle_id  = tag,
                won        = None,
                total_turns= 0,
                my_fainted = 0,
                opp_fainted= 0,
            )

        snap = self._snapshot_turn(battle)
        self._current_records[tag].turns.append(snap)

        # Pick a random move
        order = self.choose_random_move(battle)

        # Record what was chosen
        snap.action_taken = str(order)
        return order

    def _battle_finished_callback(self, battle):
        tag = battle.battle_tag
        rec = self._current_records.pop(tag, None)
        if rec is None:
            return

        rec.won         = battle.won
        rec.total_turns = battle.turn
        rec.my_fainted  = sum(p.fainted for p in battle.team.values())
        rec.opp_fainted = sum(p.fainted for p in battle.opponent_team.values())
        self.records.append(rec)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    run_id = str(int(time.time()))[-6:]

    player = TrackingPlayer(
        account_configuration=AccountConfiguration(f"tracker_{run_id}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
    )
    opponent = TrackingPlayer(
        account_configuration=AccountConfiguration(f"opponent_{run_id}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
    )

    print(f"Simulating {N_BATTLES} Gen 8 OU mirror matches...")
    await player.battle_against(opponent, n_battles=N_BATTLES)

    # ── Summary ───────────────────────────────────────────────────────────────
    records = player.records
    wins    = [r for r in records if r.won]
    losses  = [r for r in records if not r.won]

    print(f"\nResults  : {len(wins)}W / {len(losses)}L / {len(records)} total")
    print(f"Win rate : {len(wins)/len(records)*100:.1f}%")

    avg_turns = sum(r.total_turns for r in records) / len(records)
    print(f"Avg turns: {avg_turns:.1f}")

    if wins:
        print(f"\nWinning games:")
        print(f"  Avg opp fainted : {sum(r.opp_fainted for r in wins)/len(wins):.1f}")
        print(f"  Avg my fainted  : {sum(r.my_fainted  for r in wins)/len(wins):.1f}")
        print(f"  Avg turns       : {sum(r.total_turns for r in wins)/len(wins):.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    def serialise(obj):
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    output = {
        "meta": {
            "n_battles":  len(records),
            "wins":       len(wins),
            "losses":     len(losses),
            "win_rate":   len(wins) / len(records),
            "avg_turns":  avg_turns,
            "format":     BATTLE_FORMAT,
        },
        "battles": [asdict(r) for r in records],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull log saved to {OUTPUT_FILE}")
    print(f"Total turn snapshots recorded: {sum(len(r.turns) for r in records)}")


if __name__ == "__main__":
    asyncio.run(main())