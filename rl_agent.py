import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random

from poke_env.player import SinglesEnv
from poke_env.environment import SideCondition, Effect, Status
from gymnasium.spaces import Box

# ── Observation layout ────────────────────────────────────────────────────────
# Team HP×6, fainted×6, burned×6, paralyzed×6, asleep/frozen×6          (30)
# Opp  HP×6, fainted×6, burned×6, paralyzed×6, asleep/frozen×6          (30)
# My boosts×7, opp boosts×7                                              (14)
# Weather binary                                                          ( 1)
# move_eff vs active opp×4, opp_threat, move_eff_best×4                  ( 9)
# bench offensive matchup×6                                               ( 6)
# bench defensive matchup×6  [NEW]                                        ( 6)
# bench ability features×6   [NEW — regen/ironbarbs/roughskin]           ( 6)
# SR my side, SR opp side                                                 ( 2)
# my atk capped, opp atk capped                                           ( 2)
# my active one-hot×6, opp active one-hot×6                              (12)
# turn (0-1), leech seed on opp, my_locked, opp_locked                   ( 4)
# speed advantage                                                         ( 1)
# opp ability features [NEW — regen/ironbarbs/roughskin]                 ( 3)

OBS_SIZE = 126

_BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

# Abilities worth encoding explicitly
_REGEN_ABILITIES    = {"regenerator"}
_CONTACT_PUNISH     = {"ironbarbs", "roughskin"}


def _get_moves(p, opp_by_species):
    if p.moves:
        return list(p.moves.values())
    opp = opp_by_species.get(p.species)
    if opp and opp.moves:
        return list(opp.moves.values())
    return []


def _status3(pokemon):
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
        return 75
    return (pokemon.base_stats or {}).get("spe", 75)


def _ability_features(pokemon):
    """Return [is_regenerator, is_ironbarbs, is_roughskin] for a pokemon."""
    if pokemon is None:
        return [0.0, 0.0, 0.0]
    ab = (pokemon.ability or "").lower()
    return [
        float(ab in _REGEN_ABILITIES),
        float(ab == "ironbarbs"),
        float(ab == "roughskin"),
    ]


def _eff(mon, move):
    if mon is None:
        return 1.0
    try:
        return mon.damage_multiplier(move)
    except Exception:
        return 1.0


def embed_battle(battle, opp_locked_override=None) -> np.ndarray:
    my_pokemon     = list(battle.team.values())
    opp_by_species = {p.species: p for p in battle.opponent_team.values()}

    # ── Team HP / fainted ─────────────────────────────────────────────────────
    hp   = [p.current_hp_fraction for p in my_pokemon]
    fnt  = [float(p.fainted)      for p in my_pokemon]
    ohp  = [opp_by_species[p.species].current_hp_fraction
            if p.species in opp_by_species else 1.0
            for p in my_pokemon]
    ofnt = [float(opp_by_species[p.species].fainted)
            if p.species in opp_by_species else 0.0
            for p in my_pokemon]

    # ── Status ────────────────────────────────────────────────────────────────
    my_brn  = [_status3(p)[0] for p in my_pokemon]
    my_par  = [_status3(p)[1] for p in my_pokemon]
    my_slp  = [_status3(p)[2] for p in my_pokemon]
    opp_brn = [_status3(opp_by_species.get(p.species))[0] for p in my_pokemon]
    opp_par = [_status3(opp_by_species.get(p.species))[1] for p in my_pokemon]
    opp_slp = [_status3(opp_by_species.get(p.species))[2] for p in my_pokemon]

    # ── Boosts ────────────────────────────────────────────────────────────────
    ab = battle.active_pokemon.boosts if battle.active_pokemon else {}
    ob = battle.opponent_active_pokemon.boosts if battle.opponent_active_pokemon else {}
    boosts  = [ab.get(k, 0) / 6 for k in _BOOST_KEYS]
    oboosts = [ob.get(k, 0) / 6 for k in _BOOST_KEYS]

    # ── Weather ───────────────────────────────────────────────────────────────
    weather = [float(battle.weather is not None)]

    opp        = battle.opponent_active_pokemon
    alive_opps = [p for p in battle.opponent_team.values() if not p.fainted]

    # ── Move effectiveness (action indices 6-9) ───────────────────────────────
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

    # ── Opponent threat ───────────────────────────────────────────────────────
    if opp and opp.moves:
        opp_best = max(_eff(battle.active_pokemon, m) for m in opp.moves.values())
    else:
        opp_best = 1.0
    opp_threat = [(opp_best - 1.0) / 3.0]

    # ── Bench offensive matchup ───────────────────────────────────────────────
    bench_offensive = []
    for p in my_pokemon:
        if p.fainted or p.active:
            bench_offensive.append(0.0)
        else:
            moves = _get_moves(p, opp_by_species)
            if moves:
                best = max(_eff(opp, m) for m in moves)
                bench_offensive.append((best - 1.0) / 3.0)
            else:
                bench_offensive.append(0.0)

    # ── Bench defensive matchup [NEW] ─────────────────────────────────────────
    # How badly can the opponent hit each benched mon with its known moves?
    bench_defensive = []
    opp_moves = list(opp.moves.values()) if opp and opp.moves else []
    for p in my_pokemon:
        if p.fainted or p.active:
            bench_defensive.append(0.0)
        else:
            if opp_moves:
                worst = max(_eff(p, m) for m in opp_moves)
                bench_defensive.append((worst - 1.0) / 3.0)
            else:
                bench_defensive.append(0.0)

    # ── Bench ability features [NEW] ──────────────────────────────────────────
    # Encode regenerator per bench slot so the network can value regen switches
    bench_regen = []
    for p in my_pokemon:
        if p.fainted or p.active:
            bench_regen.append(0.0)
        else:
            ab_str = (p.ability or "").lower()
            bench_regen.append(float(ab_str in _REGEN_ABILITIES))

    # ── Hazards ───────────────────────────────────────────────────────────────
    sr_up  = [float(SideCondition.STEALTH_ROCK in battle.side_conditions)]
    osr_up = [float(SideCondition.STEALTH_ROCK in battle.opponent_side_conditions)]

    # ── Boost caps ────────────────────────────────────────────────────────────
    atk_capped  = [float(ab.get("atk", 0) >= 6)]
    oatk_capped = [float(ob.get("atk", 0) >= 6)]

    # ── Active one-hots ───────────────────────────────────────────────────────
    my_active = [float(p.active) for p in my_pokemon]
    opp_active_species = (battle.opponent_active_pokemon.species
                          if battle.opponent_active_pokemon else None)
    opp_active = [float(p.species == opp_active_species) for p in my_pokemon]

    # ── Misc ──────────────────────────────────────────────────────────────────
    turn       = [min(battle.turn / 100.0, 1.0)]
    leech_seed = [float(opp is not None and Effect.LEECH_SEED in opp.effects)]
    my_locked  = [float(len(battle.available_moves) == 1)]

    if opp_locked_override is not None:
        opp_locked = [float(opp_locked_override)]
    else:
        opp_locked = [float(opp is not None and len(getattr(opp, "moves", {})) == 1)]

    spe_adv = [(_base_spe(battle.active_pokemon) - _base_spe(opp)) / 100.0]

    # ── Opponent ability features [NEW] ───────────────────────────────────────
    opp_ability = _ability_features(opp)  # [regen, ironbarbs, roughskin]

    return np.array(
        hp + fnt + my_brn + my_par + my_slp
        + ohp + ofnt + opp_brn + opp_par + opp_slp
        + boosts + oboosts
        + weather
        + move_eff + opp_threat + move_eff_best
        + bench_offensive
        + bench_defensive       # NEW
        + bench_regen           # NEW
        + sr_up + osr_up
        + atk_capped + oatk_capped
        + my_active + opp_active
        + turn + leech_seed + my_locked + opp_locked
        + spe_adv
        + opp_ability,          # NEW
        dtype=np.float32,
    )


# ── Dueling DQN [NEW] ─────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 512),       nn.ReLU(),
            nn.Linear(512, 256),       nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        # Dueling aggregation: Q = V + (A - mean(A))
        return v + a - a.mean(dim=-1, keepdim=True)

    @property
    def out_features(self):
        return self.advantage[-1].out_features


# ── poke-env SinglesEnv action encoding ───────────────────────────────────────
# Per the poke-env docs (SinglesEnv.action_to_order):
#   action  0      → pass
#   action  1..6   → switch to team slot (1-indexed): slot = action - 1
#   action  7..10  → use move 1..4:                    move_idx = action - 7
#   (higher ranges are mega / z-move / dynamax / tera variants, unused for gen8ou)
SWITCH_OFFSET = 1   # action = SWITCH_OFFSET + slot_index  (slot 0..5 → action 1..6)
MOVE_OFFSET   = 7   # action = MOVE_OFFSET   + move_index  (move 0..3 → action 7..10)


def _is_useless(move, battle):
    if (move.id == "swordsdance"
            and battle.active_pokemon
            and battle.active_pokemon.boosts.get("atk", 0) >= 6):
        return True
    return False


def action_mask(battle, n_actions):
    mask = np.zeros(n_actions, dtype=bool)

    # Moves: action 7+i → active_pokemon.moves[i]
    if battle.active_pokemon and battle.active_pokemon.moves:
        avail_ids = {m.id for m in battle.available_moves}
        for i, move in enumerate(battle.active_pokemon.moves.values()):
            idx = MOVE_OFFSET + i
            if i < 4 and idx < n_actions and move.id in avail_ids and not _is_useless(move, battle):
                mask[idx] = True

    # Switches: action 1+i → list(team.values())[i]
    switch_species = {p.base_species for p in battle.available_switches}
    for i, pokemon in enumerate(battle.team.values()):
        idx = SWITCH_OFFSET + i
        if i < 6 and idx < n_actions and pokemon.base_species in switch_species:
            mask[idx] = True

    # Fallback: prefer moves over switches if nothing got masked
    if not mask.any():
        if battle.active_pokemon and battle.active_pokemon.moves:
            for i, _ in enumerate(battle.active_pokemon.moves.values()):
                idx = MOVE_OFFSET + i
                if i < 4 and idx < n_actions:
                    mask[idx] = True
        if not mask.any():
            mask[0] = True  # last resort: pass, so the mask is never all-False

    return mask


def order_to_action(order, battle):
    """Inverse of action_to_order for the plain (no-gimmick) singles subset.
    Returns the integer action for a BattleOrder, or None if it can't be mapped
    (teampreview / forfeit / pass / unmatched). Move/switch are matched by id and
    base_species respectively, mirroring action_mask and the FrozenPlayer decode.
    Gimmick orders (mega/z/dynamax/tera) are mapped to their plain-move action,
    since the RL agent's mask only ever enables plain moves and switches."""
    o = getattr(order, "order", None)
    if o is None:
        return None
    # Move objects expose base_power; Pokemon objects expose base_species.
    if hasattr(o, "base_power"):
        moves = list(battle.active_pokemon.moves.values()) if battle.active_pokemon else []
        for i, m in enumerate(moves[:4]):
            if m.id == o.id:
                return MOVE_OFFSET + i
        return None
    if hasattr(o, "base_species"):
        for i, p in enumerate(list(battle.team.values())[:6]):
            if p.base_species == o.base_species:
                return SWITCH_OFFSET + i
        return None
    return None


# ── Replay buffer ─────────────────────────────────────────────────────────────

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


# ── RLEnv ─────────────────────────────────────────────────────────────────────

class RLEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_opp_species  = None
        self._prev_opp_moves    = frozenset()
        self._prev_opp_hp       = {}   # species -> hp fraction, for damage shaping

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
        # Faint + win/loss signal
        reward = self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=0.0, victory_value=30.0,
        )

        # Small step penalty to prevent 800-turn games
        reward -= 0.01

        # Opponent HP delta shaping — reward dealing damage, ignore own HP entirely
        # This avoids any Regenerator cycling incentive
        for p in battle.opponent_team.values():
            prev = self._prev_opp_hp.get(p.species, p.current_hp_fraction)
            delta = prev - p.current_hp_fraction  # positive when opponent loses HP
            if delta > 0:
                reward += delta * 1.5
            self._prev_opp_hp[p.species] = p.current_hp_fraction

        return reward

    def reset(self, seed=None, options=None):
        # SingleAgentWrapper calls env.reset(seed, options) positionally, so the
        # override must accept them positionally and forward them on.
        self._prev_opp_hp      = {}
        self._prev_opp_species = None
        self._prev_opp_moves   = frozenset()
        return super().reset(seed=seed, options=options)