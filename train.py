"""
Curriculum RL training for poke-env (SinglesEnv + SingleAgentWrapper)
---------------------------------------------------------------------
Phase 0  : train against the built-in random opponent until competent
Phase 1+ : 50% Random anchor, 50% frozen checkpoint self-play

Run:
    python train.py

Requires local Showdown server:
    cd pokemon-showdown && node pokemon-showdown start --no-security
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from poke_env.player import (
    SingleAgentWrapper, RandomPlayer, SimpleHeuristicsPlayer, MaxBasePowerPlayer,
)
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from rl_agent import (
    RLEnv, DQN, ReplayBuffer, OBS_SIZE,
    embed_battle as _embed_battle,
    action_mask, order_to_action,
    SWITCH_OFFSET, MOVE_OFFSET,
)

# ── Hyperparameters ──────────────────────────────────────────────────────────

PHASE0_BATTLES       = 500         # short Random shakeout to confirm the pipeline
#                                    works and get off random init. Skipped
#                                    entirely when a BC warm start is loaded.
SELFPLAY_BATTLES     = 2500        # battles per generation
N_GENERATIONS        = 29
VAL_BATTLES          = 150
LEARNING_RATE        = 1e-3
GAMMA                = 0.99
EPSILON_START        = 1.0
EPSILON_END          = 0.05
EPSILON_DECAY_FAST   = 0.9999      # phase 0
EPSILON_DECAY_SLOW   = 0.99999     # self-play phases
BATCH_SIZE           = 64
TARGET_UPDATE_FREQ   = 100
REPLAY_CAPACITY      = 500_000     # larger buffer
GRAD_STEPS_PER_ENV   = 3           # multiple gradient updates per env step
GRAD_WARMUP          = 10_000      # only do multi-step once buffer has this many
# Scheduled opponent curriculum (linear over generations 1..N). Weights are
# per-battle within each phase. Self-play is the remainder (1 - others), and
# only enters once a checkpoint qualifies; until then run_phase renormalizes
# the present opponents proportionally. Endpoints chosen around poke-env's
# skill ratings (Random~1, MaxBasePower~8, SimpleHeuristics~129) so the agent
# always trains against opponents near its level.
RANDOM_START, RANDOM_END = 0.40, 0.10   # easy wins early; forgetting anchor late
MAXBP_START,  MAXBP_END  = 0.35, 0.10   # the winnable middle rung
HEUR_START,   HEUR_END   = 0.10, 0.50   # ramps up to the main challenge
#                                         self-play = remainder (~0.15 → ~0.30)

# Validation split (must sum to 1.0). The self-play gate keys off MaxBasePower,
# a bar the agent can realistically clear on the way up (unlike the heuristic).
VAL_RANDOM_FRACTION  = 0.30
VAL_MAXBP_FRACTION   = 0.30
VAL_HEUR_FRACTION    = 0.40
SELFPLAY_WIN_FLOOR   = 0.35        # min win rate vs MaxBasePower to enter the
#                                    self-play opponent pool
CHECKPOINT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
BATTLE_FORMAT        = "gen8ou"
PRETRAINED_PATH      = os.path.join(CHECKPOINT_DIR, "checkpoint_v10.pt")
WARMSTART_EPSILON    = 0.30        # lower exploration if we loaded a BC warm start

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

RUN_ID = str(int(time.time()))[-4:]
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(opponent, tag1):
    env = RLEnv(
        account_configuration1=AccountConfiguration(tag1, None),
        account_configuration2=AccountConfiguration(f"{tag1}p2", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
        start_challenging=True,
        strict=False,
    )
    return SingleAgentWrapper(env, opponent)


def safe_close(env):
    """poke-env can raise in _stop_challenge_loop if the challenge loop is still
    settling when close() is called (assert self.agent2_to_move). The error is
    purely in teardown — training for the phase is already done — so we let the
    battle drain briefly, then swallow any close-time exception."""
    try:
        time.sleep(0.5)
        env.close()
    except Exception as e:
        print(f"  (ignored close() error: {type(e).__name__})")


def train_step(policy_net, target_net, optimizer, replay, device):
    if len(replay) < BATCH_SIZE:
        return None
    states, actions, rewards, next_states, dones, next_masks = replay.sample(BATCH_SIZE)
    s  = torch.FloatTensor(states).to(device)
    a  = torch.LongTensor(actions).unsqueeze(1).to(device)
    r  = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    s2 = torch.FloatTensor(next_states).to(device)
    d  = torch.FloatTensor(dones).unsqueeze(1).to(device)
    nm = torch.from_numpy(np.asarray(next_masks)).bool().to(device)  # (B, n_actions)

    q = policy_net(s).gather(1, a)
    with torch.no_grad():
        # Double DQN with action masking: both the policy net's action selection
        # AND the target net's value lookup must ignore illegal next-state actions.
        # Without this, illegal actions can carry arbitrary (untrained) Q-values
        # that poison the bootstrap target and corrupt the greedy policy.
        next_q_policy = policy_net(s2).clone()
        next_q_policy[~nm] = float('-inf')
        next_actions = next_q_policy.argmax(1, keepdim=True)

        next_q_target = target_net(s2).clone()
        next_q_target[~nm] = float('-inf')
        nq = next_q_target.gather(1, next_actions)
        # If a row were fully masked (shouldn't happen), guard against -inf
        nq = torch.nan_to_num(nq, neginf=0.0)

        tg = r + GAMMA * nq * (1 - d)
    loss = nn.SmoothL1Loss()(q, tg)  # Huber loss is more stable than MSE
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def select_action(obs, policy_net, n_actions, epsilon, device, mask=None):
    if random.random() < epsilon:
        valid = np.where(mask)[0] if mask is not None else np.arange(n_actions)
        return np.random.choice(valid)
    with torch.no_grad():
        q = policy_net(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze()
        if mask is not None:
            q = q.clone()
            q[~torch.from_numpy(mask).to(device)] = float('-inf')
        return np.int64(q.argmax().item())


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_against(opp, tag, n_battles, policy_net, n_actions, device):
    """Run n_battles greedy (epsilon=0) battles vs a given opponent; return wins."""
    val_env = make_env(opp, tag)
    wins = 0
    for _ in range(n_battles):
        obs, _ = val_env.reset()
        done = truncated = False
        while not (done or truncated):
            try:
                mask = action_mask(val_env.env.battle1, n_actions)
            except AttributeError:
                mask = None
            action = select_action(obs, policy_net, n_actions, 0.0, device, mask)
            obs, reward, done, truncated, _ = val_env.step(action)
        try:
            if val_env.env.battle1.won:
                wins += 1
        except AttributeError:
            pass
    safe_close(val_env)
    return wins


def run_validation(policy_net, n_actions, device, gen):
    # Split the validation budget across Random, MaxBasePower, and the heuristic.
    n_random = int(round(VAL_BATTLES * VAL_RANDOM_FRACTION))
    n_maxbp  = int(round(VAL_BATTLES * VAL_MAXBP_FRACTION))
    n_heur   = VAL_BATTLES - n_random - n_maxbp

    print(f"\n  [ Validation — {n_random} vs Random, {n_maxbp} vs MaxBasePower, "
          f"{n_heur} vs SimpleHeuristics ]")

    rand_opp = RandomPlayer(
        account_configuration=AccountConfiguration(f"valrandom{RUN_ID}g{gen}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT, team=TEAM,
    )
    maxbp_opp = MaxBasePowerPlayer(
        account_configuration=AccountConfiguration(f"valmaxbp{RUN_ID}g{gen}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT, team=TEAM,
    )
    heur_opp = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"valheur{RUN_ID}g{gen}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT, team=TEAM,
    )

    rand_wins  = _validate_against(rand_opp,  f"valagentr{RUN_ID}g{gen}",
                                   n_random, policy_net, n_actions, device)
    maxbp_wins = _validate_against(maxbp_opp, f"valagentm{RUN_ID}g{gen}",
                                   n_maxbp, policy_net, n_actions, device)
    heur_wins  = _validate_against(heur_opp,  f"valagenth{RUN_ID}g{gen}",
                                   n_heur, policy_net, n_actions, device)

    rand_rate  = rand_wins / n_random if n_random else float('nan')
    maxbp_rate = maxbp_wins / n_maxbp if n_maxbp else float('nan')
    heur_rate  = heur_wins / n_heur if n_heur else float('nan')
    combined   = (rand_wins + maxbp_wins + heur_wins) / VAL_BATTLES

    print(f"  vs Random:          {rand_rate*100:5.1f}%  ({rand_wins}/{n_random})")
    print(f"  vs MaxBasePower:    {maxbp_rate*100:5.1f}%  ({maxbp_wins}/{n_maxbp})")
    print(f"  vs SimpleHeuristics:{heur_rate*100:5.1f}%  ({heur_wins}/{n_heur})")
    print(f"  combined:           {combined*100:5.1f}%\n")

    # Gate self-play eligibility on MaxBasePower — a rung the agent can clear on
    # the way up. (The heuristic rate is the real progress metric to watch.)
    return maxbp_rate


# ── Training phase ────────────────────────────────────────────────────────────

def _run_one_battle(env, policy_net, optimizer, target_net, replay, device,
                    epsilon, step, decay):
    """Play one battle on the given env, pushing transitions and training.
    Returns (won, episode_reward)."""
    obs, _ = env.reset()
    done = truncated = False
    ep_reward = 0.0
    n_actions = env.action_space.n

    try:
        mask = action_mask(env.env.battle1, n_actions)
    except AttributeError:
        mask = np.ones(n_actions, dtype=bool)

    while not (done or truncated):
        action = select_action(obs, policy_net, n_actions, epsilon[0], device, mask)
        next_obs, reward, done, truncated, _ = env.step(action)

        if done or truncated:
            next_mask = np.ones(n_actions, dtype=bool)
        else:
            try:
                next_mask = action_mask(env.env.battle1, n_actions)
            except AttributeError:
                next_mask = np.ones(n_actions, dtype=bool)

        replay.push(obs, action, reward, next_obs,
                    float(done or truncated), next_mask)

        n_grad = GRAD_STEPS_PER_ENV if len(replay) >= GRAD_WARMUP else 1
        for _ in range(n_grad):
            train_step(policy_net, target_net, optimizer, replay, device)

        step[0] += 1
        if step[0] % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon[0] = max(EPSILON_END, epsilon[0] * decay)
        ep_reward += reward
        obs = next_obs
        mask = next_mask

    won = False
    try:
        won = bool(env.env.battle1.won)
    except AttributeError:
        pass
    return won, ep_reward


def run_phase(weighted_envs, policy_net, target_net, optimizer, replay, device,
              n_battles, epsilon, step, decay, label):
    """weighted_envs: list of (name, env, weight). Each battle samples one env by
    weight, so every phase contains the full opponent mix (Random anchor always
    present). Per-opponent win rates are tracked separately and reported."""
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    names   = [w[0] for w in weighted_envs]
    envs    = [w[1] for w in weighted_envs]
    weights = np.array([w[2] for w in weighted_envs], dtype=float)
    weights /= weights.sum()

    wins = total_reward = 0
    recent_wins = []
    per_opp = {n: [0, 0] for n in names}  # name -> [wins, battles]

    for i in range(n_battles):
        j = int(np.random.choice(len(envs), p=weights))
        won, ep_reward = _run_one_battle(
            envs[j], policy_net, optimizer, target_net, replay, device,
            epsilon, step, decay)

        total_reward += ep_reward
        wins += int(won)
        per_opp[names[j]][0] += int(won)
        per_opp[names[j]][1] += 1
        recent_wins.append(int(won))
        if len(recent_wins) > 200:
            recent_wins.pop(0)

        if (i + 1) % 50 == 0:
            print(f"  Battle {i+1:>4}/{n_battles} | "
                  f"win rate {wins/(i+1)*100:5.1f}% | ε={epsilon[0]:.3f}")

    trailing = sum(recent_wins) / len(recent_wins) if recent_wins else 0.0
    print(f"\n  Final win rate: {wins/n_battles*100:.1f}%  "
          f"trailing-200: {trailing*100:.1f}%  "
          f"avg reward: {total_reward/n_battles:.2f}")
    # Per-opponent breakdown so you can see e.g. win rate vs Random specifically.
    for n in names:
        w_, b_ = per_opp[n]
        if b_:
            print(f"    vs {n:<14} {w_/b_*100:5.1f}%  ({w_}/{b_})")

    for e in envs:
        safe_close(e)
    return trailing


# ── Frozen greedy opponent ────────────────────────────────────────────────────

class FrozenPlayer(RandomPlayer):
    def __init__(self, net, device, n_actions, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.dev = device
        self.n_actions = n_actions

    def choose_move(self, battle):
        obs = _embed_battle(battle)
        n_actions = self.n_actions
        mask = action_mask(battle, n_actions)
        with torch.no_grad():
            q = self.net(torch.FloatTensor(obs).unsqueeze(0).to(self.dev)).squeeze()
            q = q.clone()
            q[~torch.from_numpy(mask).to(self.dev)] = float('-inf')
        action = int(q.argmax().item())

        # Decode using the real poke-env encoding (switch 1..6, move 7..10)
        if SWITCH_OFFSET <= action < SWITCH_OFFSET + 6:
            slot = action - SWITCH_OFFSET
            team = list(battle.team.values())
            switch_species = {p.base_species for p in battle.available_switches}
            if slot < len(team) and team[slot].base_species in switch_species:
                return self.create_order(team[slot])
        elif MOVE_OFFSET <= action < MOVE_OFFSET + 4:
            move_idx = action - MOVE_OFFSET
            if battle.active_pokemon and battle.active_pokemon.moves:
                all_moves = list(battle.active_pokemon.moves.values())
                avail_ids = {m.id for m in battle.available_moves}
                if move_idx < len(all_moves) and all_moves[move_idx].id in avail_ids:
                    return self.create_order(all_moves[move_idx])
        return self.choose_random_move(battle)


# ── Opponent factory ──────────────────────────────────────────────────────────

def make_random_opp(tag):
    return RandomPlayer(
        account_configuration=AccountConfiguration(tag, None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
    )


def make_heuristic_opp(tag):
    return SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(tag, None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
    )


def make_maxbp_opp(tag):
    return MaxBasePowerPlayer(
        account_configuration=AccountConfiguration(tag, None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
    )


def curriculum_weights(gen, n_gens):
    """Linear interpolation of opponent weights across the curriculum.
    Returns (w_random, w_maxbp, w_heur, w_selfplay) summing to 1.0.
    Self-play is whatever is left after the three scripted opponents."""
    t = (gen - 1) / max(1, n_gens - 1)   # 0.0 at gen 1, 1.0 at gen N
    w_random = RANDOM_START + t * (RANDOM_END - RANDOM_START)
    w_maxbp  = MAXBP_START  + t * (MAXBP_END  - MAXBP_START)
    w_heur   = HEUR_START   + t * (HEUR_END   - HEUR_START)
    w_self   = max(0.0, 1.0 - w_random - w_maxbp - w_heur)
    return w_random, w_maxbp, w_heur, w_self


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run ID: {RUN_ID}")

    # Probe action space with a throwaway env
    probe_opp = make_random_opp(f"probe{RUN_ID}")
    probe_env = make_env(probe_opp, f"probeagent{RUN_ID}")
    n_actions = probe_env.action_space.n
    safe_close(probe_env)
    print(f"Action space size: {n_actions}")

    policy_net = DQN(OBS_SIZE, n_actions).to(device)
    target_net = DQN(OBS_SIZE, n_actions).to(device)

    # Behavior-cloning warm start: if pretrain.py produced weights, load them so
    # the agent begins from a competent policy instead of random. Lower the
    # starting epsilon accordingly — we want to exploit the warm start, not bury
    # it under random exploration.
    start_epsilon = EPSILON_START
    warm_started = False
    if os.path.exists(PRETRAINED_PATH):
        try:
            policy_net.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
            start_epsilon = WARMSTART_EPSILON
            warm_started = True
            print(f"Loaded BC warm start from {PRETRAINED_PATH} "
                  f"(starting ε={start_epsilon})")
        except Exception as e:
            print(f"Could not load pretrained weights ({type(e).__name__}: {e}); "
                  f"starting from scratch.")
    else:
        print("No pretrained weights found; starting from scratch.")

    target_net.load_state_dict(policy_net.state_dict())
    optimizer  = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay     = ReplayBuffer(capacity=REPLAY_CAPACITY)
    epsilon    = [start_epsilon]
    step       = [0]

    # ── Phase 0: short Random shakeout (skipped if BC warm start loaded) ───────
    # With a warm start the network already plays competently, so relearning the
    # basics against Random would just waste battles. Without one, a short pass
    # vs Random gets the network off random init and confirms the action
    # pipeline works before committing to the real curriculum.
    if warm_started:
        print("\n  Warm start loaded — skipping Random phase 0, "
              "saving it directly as checkpoint_v0.")
    else:
        opp0 = make_random_opp(f"random{RUN_ID}p0")
        env0 = make_env(opp0, f"agent{RUN_ID}p0")
        phase0_rate = run_phase([("Random", env0, 1.0)],
                                policy_net, target_net, optimizer, replay,
                                device, PHASE0_BATTLES, epsilon, step,
                                EPSILON_DECAY_FAST,
                                "Phase 0 — Random shakeout")
        print(f"\n  Phase 0 win rate vs Random: {phase0_rate*100:.1f}%")
        if phase0_rate < 0.30:
            print("  ⚠ Win rate below 30% vs Random suggests a pipeline bug "
                  "(action encoding, masking, or observation). Investigate "
                  "before trusting the rest of the run.")

    ckpt = os.path.join(CHECKPOINT_DIR, "checkpoint_v0.pt")
    torch.save(policy_net.state_dict(), ckpt)
    print(f"\n  ✓ Saved {ckpt}")
    last_val = run_validation(policy_net, n_actions, device, 0)

    # Track which checkpoints are strong enough to self-play against
    strong_checkpoints = [0] if last_val >= SELFPLAY_WIN_FLOOR else []

    # ── Scheduled curriculum ──────────────────────────────────────────────────
    # Each phase trains against a MIX sampled per battle. Weights ramp linearly
    # across generations: Random and MaxBasePower start high and decay, the
    # heuristic ramps up, self-play grows as the remainder. Every phase always
    # contains the easy anchors so there's never a phase with no winnable games.
    for gen in range(1, N_GENERATIONS + 1):
        w_random, w_maxbp, w_heur, w_self = curriculum_weights(gen, N_GENERATIONS)

        weighted = [
            ("Random", make_env(make_random_opp(f"random{RUN_ID}g{gen}"),
                                f"agentr{RUN_ID}g{gen}"), w_random),
            ("MaxBasePower", make_env(make_maxbp_opp(f"maxbp{RUN_ID}g{gen}"),
                                      f"agentm{RUN_ID}g{gen}"), w_maxbp),
            ("Heuristic", make_env(make_heuristic_opp(f"heuristic{RUN_ID}g{gen}"),
                                   f"agenth{RUN_ID}g{gen}"), w_heur),
        ]

        # Self-play slot: only once a checkpoint qualifies. If none has, its
        # weight is simply absent and run_phase renormalizes the scripted
        # opponents proportionally.
        if strong_checkpoints and w_self > 0:
            sw = np.arange(1, len(strong_checkpoints) + 1, dtype=float)
            sw /= sw.sum()
            chosen = int(np.random.choice(strong_checkpoints, p=sw))
            chosen_ckpt = os.path.join(CHECKPOINT_DIR, f"checkpoint_v{chosen}.pt")
            frozen_net = DQN(OBS_SIZE, n_actions).to(device)
            frozen_net.load_state_dict(torch.load(chosen_ckpt, map_location=device))
            frozen_net.eval()
            sp_opp = FrozenPlayer(
                net=frozen_net, device=device, n_actions=n_actions,
                account_configuration=AccountConfiguration(f"selfv{chosen}n{RUN_ID}g{gen}", None),
                server_configuration=LocalhostServerConfiguration,
                battle_format=BATTLE_FORMAT,
                team=TEAM,
            )
            weighted.append(
                (f"SelfPlay-v{chosen}",
                 make_env(sp_opp, f"agents{RUN_ID}g{gen}"), w_self))
            sp_note = f"self-play v{chosen} {w_self:.0%}"
        else:
            sp_note = "no self-play yet"

        label = (f"Phase {gen} — R {w_random:.0%} / MaxBP {w_maxbp:.0%} / "
                 f"Heur {w_heur:.0%} / {sp_note}")

        run_phase(weighted, policy_net, target_net, optimizer, replay, device,
                  SELFPLAY_BATTLES, epsilon, step, EPSILON_DECAY_SLOW, label)

        ckpt = os.path.join(CHECKPOINT_DIR, f"checkpoint_v{gen}.pt")
        torch.save(policy_net.state_dict(), ckpt)
        print(f"\n  ✓ Saved {ckpt}")
        val_rate = run_validation(policy_net, n_actions, device, gen)

        # Enter the self-play pool once it can beat MaxBasePower reliably
        if val_rate >= SELFPLAY_WIN_FLOOR:
            strong_checkpoints.append(gen)

    print(f"\n\nCurriculum complete! Final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()