import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

from poke_env.player import SinglesEnv
from poke_env.environment import SideCondition, Effect, Status
from gymnasium.spaces import Box

OBS_SIZE = 111
# Layout (all values in [-1, 1] unless noted):
#   Team HP×6, fainted×6, burned×6, paralyzed×6, asleep/frozen×6          (30)
#   Opp  HP×6, fainted×6, burned×6, paralyzed×6, asleep/frozen×6          (30)
#   My boosts×7, opp boosts×7                                              (14)
#   Weather binary                                                          ( 1)
#   move_eff vs active opp×4, opp_threat, move_eff_best×4                  ( 9)
#   bench matchup×6                                                         ( 6)
#   SR my side, SR opp side                                                 ( 2)
#   my atk capped, opp atk capped                                           ( 2)
#   my active one-hot×6, opp active one-hot×6                              (12)
#   turn (0-1), leech seed on opp, my_locked, opp_locked                   ( 4)
#   speed advantage (my base spe - opp base spe) / 100                     ( 1)

_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def _get_moves(p, opp_by_species):
    """Return move list for p, using opponent's copy if ours hasn't been revealed yet."""
    if p.moves:
        return list(p.moves.values())
    opp = opp_by_species.get(p.species)
    if opp and opp.moves:
        return list(opp.moves.values())
    return []


def _status3(pokemon):
    """Return [burned, paralyzed, asleep_or_frozen] for a pokemon (or zeros if None)."""
    if pokemon is None:
        return [0.0, 0.0, 0.0]
    s = pokemon.status
    return [
        float(s == Status.BRN),
        float(s == Status.PAR),
        float(s in (Status.SLP, Status.FRZ)),
    ]


def _base_spe(pokemon):
    if pokemon is None:
        return 75  # rough midpoint
    return (pokemon.base_stats or {}).get("spe", 75)


def embed_battle(battle, opp_locked_override=None) -> np.ndarray:
    my_pokemon     = list(battle.team.values())
    opp_by_species = {p.species: p for p in battle.opponent_team.values()}

    # ── Team HP / fainted ──────────────────────────────────────────────────────
    hp   = [p.current_hp_fraction for p in my_pokemon]
    fnt  = [float(p.fainted)      for p in my_pokemon]
    ohp  = [opp_by_species[p.species].current_hp_fraction if p.species in opp_by_species else 1.0
            for p in my_pokemon]
    ofnt = [float(opp_by_species[p.species].fainted) if p.species in opp_by_species else 0.0
            for p in my_pokemon]

    # ── Status (3 bits per mon: burned / paralyzed / asleep-frozen) ───────────
    my_brn  = [_status3(p)[0] for p in my_pokemon]
    my_par  = [_status3(p)[1] for p in my_pokemon]
    my_slp  = [_status3(p)[2] for p in my_pokemon]
    opp_brn = [_status3(opp_by_species.get(p.species))[0] for p in my_pokemon]
    opp_par = [_status3(opp_by_species.get(p.species))[1] for p in my_pokemon]
    opp_slp = [_status3(opp_by_species.get(p.species))[2] for p in my_pokemon]

    # ── Boosts ─────────────────────────────────────────────────────────────────
    ab = battle.active_pokemon.boosts if battle.active_pokemon else {}
    ob = battle.opponent_active_pokemon.boosts if battle.opponent_active_pokemon else {}
    boosts  = [ab.get(k, 0) / 6 for k in _BOOST_KEYS]
    oboosts = [ob.get(k, 0) / 6 for k in _BOOST_KEYS]

    # ── Weather ────────────────────────────────────────────────────────────────
    weather = [float(battle.weather is not None)]

    opp        = battle.opponent_active_pokemon
    alive_opps = [p for p in battle.opponent_team.values() if not p.fainted]

    def _eff(mon, move):
        try:
            return mon.damage_multiplier(move)
        except Exception:
            return 1.0

    # ── Move effectiveness — aligned with action indices 6-9 ──────────────────
    # Use active_pokemon.moves order (matches poke-env's action_to_order).
    # available_moves is the playable subset; 0.0 for locked/disabled slots.
    all_moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
    avail_ids = {m.id for m in battle.available_moves}

    move_eff = []
    for m in all_moves[:4]:
        move_eff.append((_eff(opp, m) - 1.0) / 3.0 if m.id in avail_ids else 0.0)
    while len(move_eff) < 4:
        move_eff.append(0.0)

    move_eff_best = []
    for m in all_moves[:4]:
        if m.id in avail_ids:
            best = max((_eff(p, m) for p in alive_opps), default=1.0)
            move_eff_best.append((best - 1.0) / 3.0)
        else:
            move_eff_best.append(0.0)
    while len(move_eff_best) < 4:
        move_eff_best.append(0.0)

    # ── Opponent threat (best move effectiveness against my active) ────────────
    if opp and opp.moves:
        opp_best = max(_eff(battle.active_pokemon, m) for m in opp.moves.values())
    else:
        opp_best = 1.0
    opp_threat = [(opp_best - 1.0) / 3.0]

    # ── Bench matchup (best move effectiveness per benched mon vs current opp) ─
    bench_matchup = []
    for p in my_pokemon:
        if p.fainted or p.active:
            bench_matchup.append(0.0)
        else:
            moves = _get_moves(p, opp_by_species)
            if moves:
                best = max(_eff(opp, m) for m in moves)
                bench_matchup.append((best - 1.0) / 3.0)
            else:
                bench_matchup.append(0.0)

    # ── Hazards ────────────────────────────────────────────────────────────────
    sr_up  = [float(SideCondition.STEALTH_ROCK in battle.side_conditions)]
    osr_up = [float(SideCondition.STEALTH_ROCK in battle.opponent_side_conditions)]

    # ── Boost caps ────────────────────────────────────────────────────────────
    atk_capped  = [float(ab.get("atk", 0) >= 6)]
    oatk_capped = [float(ob.get("atk", 0) >= 6)]

    # ── Active one-hots ────────────────────────────────────────────────────────
    my_active = [float(p.active) for p in my_pokemon]
    opp_active_species = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else None
    opp_active = [float(p.species == opp_active_species) for p in my_pokemon]

    # ── Misc ───────────────────────────────────────────────────────────────────
    turn       = [min(battle.turn / 100.0, 1.0)]
    leech_seed = [float(opp is not None and Effect.LEECH_SEED in opp.effects)]

    # my_locked: only 1 move available (choice lock, Encore, etc.)
    my_locked  = [float(len(battle.available_moves) == 1)]

    # opp_locked: stateful override from RLEnv (opponent repeated a seen move).
    # Fallback for FrozenPlayer: opponent has only been seen using 1 distinct move.
    if opp_locked_override is not None:
        opp_locked = [float(opp_locked_override)]
    else:
        opp_locked = [float(opp is not None and len(getattr(opp, "moves", {})) == 1)]

    # Speed advantage: (my base spe − opp base spe) / 100.
    # Encodes speed tier difference; paralysis features let the net adjust for that penalty.
    spe_adv = [(_base_spe(battle.active_pokemon) - _base_spe(opp)) / 100.0]

    return np.array(
        hp + fnt + my_brn + my_par + my_slp
        + ohp + ofnt + opp_brn + opp_par + opp_slp
        + boosts + oboosts
        + weather
        + move_eff + opp_threat + move_eff_best
        + bench_matchup
        + sr_up + osr_up
        + atk_capped + oatk_capped
        + my_active + opp_active
        + turn + leech_seed + my_locked + opp_locked
        + spe_adv,
        dtype=np.float32,
    )


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)


class RLEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_opp_species = None
        self._prev_opp_moves   = frozenset()

    @property
    def observation_spaces(self):
        return {
            agent: Box(
                low=-np.ones(OBS_SIZE, dtype=np.float32),
                high=np.ones(OBS_SIZE, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def embed_battle(self, battle):
        opp     = battle.opponent_active_pokemon
        species = opp.species if opp else None
        moves   = frozenset(opp.moves) if opp else frozenset()

        # True if opponent is actively repeating a seen move — best proxy for choice lock
        opp_locked = (
            species is not None
            and species == self._prev_opp_species
            and len(moves) > 0
            and moves == self._prev_opp_moves
        )
        self._prev_opp_species = species
        self._prev_opp_moves   = moves
        return embed_battle(battle, opp_locked)

    def calc_reward(self, battle):
        reward = self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=0.0, victory_value=30.0,
        )
        return reward - 0.05
