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

from poke_env.player import SingleAgentWrapper, RandomPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from rl_agent import RLEnv, DQN, ReplayBuffer, OBS_SIZE, embed_battle as _embed_battle

# ── Hyperparameters ──────────────────────────────────────────────────────────

PHASE0_BATTLES       = 10000        # much longer first phase
PHASE0_MAX_LOOPS     = 4           # loop phase 0 until win rate threshold met
PHASE0_WIN_THRESHOLD = 0.55        # don't freeze v0 below this
SELFPLAY_BATTLES     = 25000        # battles per self-play phase
N_GENERATIONS        = 10
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
RANDOM_SLOT_PROB     = 0.50        # keep Random at 50% throughout
SELFPLAY_WIN_FLOOR   = 0.55        # don't self-play vs a checkpoint below this
CHECKPOINT_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
BATTLE_FORMAT        = "gen8ou"

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

RUN_ID = str(int(time.time()))[-6:]
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


def _is_useless(move, battle):
    if (move.id == "swordsdance"
            and battle.active_pokemon.boosts.get("atk", 0) >= 6):
        return True
    return False


def action_mask(battle, n_actions):
    mask = np.zeros(n_actions, dtype=bool)

    # Moves first (action 6+i → active_pokemon.moves[i])
    if battle.active_pokemon and battle.active_pokemon.moves:
        avail_ids = {m.id for m in battle.available_moves}
        for i, move in enumerate(battle.active_pokemon.moves.values()):
            if i < 4 and move.id in avail_ids and not _is_useless(move, battle):
                mask[6 + i] = True

    # Switches (action i → list(team.values())[i])
    switch_species = {p.base_species for p in battle.available_switches}
    for i, pokemon in enumerate(battle.team.values()):
        if i < 6 and pokemon.base_species in switch_species:
            mask[i] = True

    # Fallback: prefer moves over switches if nothing got masked
    if not mask.any():
        if battle.active_pokemon and battle.active_pokemon.moves:
            for i, _ in enumerate(battle.active_pokemon.moves.values()):
                if i < 4:
                    mask[6 + i] = True
        if not mask.any():
            mask[:] = True

    return mask


def train_step(policy_net, target_net, optimizer, replay, device):
    if len(replay) < BATCH_SIZE:
        return None
    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)
    s  = torch.FloatTensor(states).to(device)
    a  = torch.LongTensor(actions).unsqueeze(1).to(device)
    r  = torch.FloatTensor(rewards).unsqueeze(1).to(device)
    s2 = torch.FloatTensor(next_states).to(device)
    d  = torch.FloatTensor(dones).unsqueeze(1).to(device)

    q = policy_net(s).gather(1, a)
    with torch.no_grad():
        # Double DQN: action selection from policy net, value from target net
        next_actions = policy_net(s2).argmax(1, keepdim=True)
        nq = target_net(s2).gather(1, next_actions)
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

def run_validation(policy_net, n_actions, device, gen):
    print(f"\n  [ Validation vs Random — {VAL_BATTLES} battles ]")
    val_opp = RandomPlayer(
        account_configuration=AccountConfiguration(f"valopp{RUN_ID}g{gen}", None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=BATTLE_FORMAT,
        team=TEAM,
    )
    val_env = make_env(val_opp, f"validbot{RUN_ID}g{gen}")
    wins = 0
    for _ in range(VAL_BATTLES):
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
    rate = wins / VAL_BATTLES
    print(f"  Validation win rate: {rate*100:.1f}%\n")
    return rate


# ── Training phase ────────────────────────────────────────────────────────────

def run_phase(env, policy_net, target_net, optimizer, replay, device,
              n_battles, epsilon, step, decay, label):
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    wins = total_reward = 0
    recent_wins = []  # rolling window for the trailing-200 win rate

    for i in range(n_battles):
        obs, _ = env.reset()
        done = truncated = False
        ep_reward = 0.0

        while not (done or truncated):
            n_actions = env.action_space.n
            try:
                mask = action_mask(env.env.battle1, n_actions)
            except AttributeError:
                mask = None
            action = select_action(obs, policy_net, n_actions, epsilon[0], device, mask)
            next_obs, reward, done, truncated, _ = env.step(action)

            replay.push(obs, action, reward, next_obs, float(done or truncated))

            # Multiple gradient steps once buffer is warm
            n_grad = GRAD_STEPS_PER_ENV if len(replay) >= GRAD_WARMUP else 1
            for _ in range(n_grad):
                train_step(policy_net, target_net, optimizer, replay, device)

            step[0] += 1
            if step[0] % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            epsilon[0] = max(EPSILON_END, epsilon[0] * decay)
            ep_reward += reward
            obs = next_obs

        total_reward += ep_reward
        won = False
        try:
            won = bool(env.env.battle1.won)
        except AttributeError:
            pass
        wins += int(won)
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
    safe_close(env)
    return trailing


# ── Frozen greedy opponent ────────────────────────────────────────────────────

class FrozenPlayer(RandomPlayer):
    def __init__(self, net, device, **kwargs):
        super().__init__(**kwargs)
        self.net = net
        self.dev = device

    def choose_move(self, battle):
        obs = _embed_battle(battle)
        n_actions = self.net.out_features
        mask = action_mask(battle, n_actions)
        with torch.no_grad():
            q = self.net(torch.FloatTensor(obs).unsqueeze(0).to(self.dev)).squeeze()
            q = q.clone()
            q[~torch.from_numpy(mask).to(self.dev)] = float('-inf')
        action = q.argmax().item()

        if action < 6:
            team = list(battle.team.values())
            switch_species = {p.base_species for p in battle.available_switches}
            if action < len(team) and team[action].base_species in switch_species:
                return self.create_order(team[action])
        else:
            move_idx = (action - 6) % 4
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run ID: {RUN_ID}")

    # Probe action space with a throwaway env
    probe_opp = make_random_opp(f"probe{RUN_ID}")
    probe_env = make_env(probe_opp, f"probebot{RUN_ID}")
    n_actions = probe_env.action_space.n
    safe_close(probe_env)
    print(f"Action space size: {n_actions}")

    policy_net = DQN(OBS_SIZE, n_actions).to(device)
    target_net = DQN(OBS_SIZE, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer  = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay     = ReplayBuffer(capacity=REPLAY_CAPACITY)
    epsilon    = [EPSILON_START]
    step       = [0]

    # ── Phase 0: loop vs Random until competent ───────────────────────────────
    phase0_rate = 0.0
    for loop in range(PHASE0_MAX_LOOPS):
        opp0 = make_random_opp(f"randomopp{RUN_ID}l{loop}")
        env0 = make_env(opp0, f"rlagent{RUN_ID}l{loop}")
        label = f"Phase 0 (loop {loop+1}/{PHASE0_MAX_LOOPS}) — training vs RandomPlayer"
        phase0_rate = run_phase(env0, policy_net, target_net, optimizer, replay,
                                device, PHASE0_BATTLES, epsilon, step,
                                EPSILON_DECAY_FAST, label)
        if phase0_rate >= PHASE0_WIN_THRESHOLD:
            print(f"\n  Phase 0 threshold reached ({phase0_rate*100:.1f}%).")
            break
        print(f"\n  Phase 0 below threshold ({phase0_rate*100:.1f}%), looping…")

    ckpt = os.path.join(CHECKPOINT_DIR, "checkpoint_v0.pt")
    torch.save(policy_net.state_dict(), ckpt)
    print(f"\n  ✓ Saved {ckpt}")
    last_val = run_validation(policy_net, n_actions, device, 0)

    # Track which checkpoints are strong enough to self-play against
    strong_checkpoints = [0] if last_val >= SELFPLAY_WIN_FLOOR else []

    # ── Self-play curriculum ──────────────────────────────────────────────────
    for gen in range(1, N_GENERATIONS + 1):
        use_random = (random.random() < RANDOM_SLOT_PROB) or (not strong_checkpoints)

        if use_random:
            opp = make_random_opp(f"rndopp{RUN_ID}g{gen}")
            env_gen = make_env(opp, f"rlagent{RUN_ID}g{gen}")
            label = f"Phase {gen} — Random anchor (50% slot)"
        else:
            # Sample from strong checkpoints, weighted toward newer
            w = np.arange(1, len(strong_checkpoints) + 1, dtype=float)
            w /= w.sum()
            chosen = int(np.random.choice(strong_checkpoints, p=w))
            chosen_ckpt = os.path.join(CHECKPOINT_DIR, f"checkpoint_v{chosen}.pt")
            frozen_net = DQN(OBS_SIZE, n_actions).to(device)
            frozen_net.load_state_dict(torch.load(chosen_ckpt, map_location=device))
            frozen_net.eval()
            opp = FrozenPlayer(
                net=frozen_net, device=device,
                account_configuration=AccountConfiguration(f"frz{RUN_ID}v{chosen}g{gen}", None),
                server_configuration=LocalhostServerConfiguration,
                battle_format=BATTLE_FORMAT,
                team=TEAM,
            )
            env_gen = make_env(opp, f"rlagent{RUN_ID}g{gen}")
            label = f"Phase {gen} — self-play vs checkpoint v{chosen}"

        run_phase(env_gen, policy_net, target_net, optimizer, replay, device,
                  SELFPLAY_BATTLES, epsilon, step, EPSILON_DECAY_SLOW, label)

        ckpt = os.path.join(CHECKPOINT_DIR, f"checkpoint_v{gen}.pt")
        torch.save(policy_net.state_dict(), ckpt)
        print(f"\n  ✓ Saved {ckpt}")
        val_rate = run_validation(policy_net, n_actions, device, gen)

        # Only add to the self-play pool if it's strong enough
        if val_rate >= SELFPLAY_WIN_FLOOR:
            strong_checkpoints.append(gen)

    print(f"\n\nCurriculum complete! Final checkpoint: {ckpt}")


if __name__ == "__main__":
    main()