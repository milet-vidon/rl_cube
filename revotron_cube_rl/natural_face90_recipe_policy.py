"""State PPO specialist; inspired by Lin et al., CoRL 2025, not a reproduction.

The deployed action is still a canonical 56-joint target. Only the right 28
joints are learned in this first specialist; R27/presentation support is frozen.
No simulator, camera, synthetic contact or pose-writing code lives here.
"""
import math
import torch
from torch import nn

SCHEMA = 'natural_face90_recipe_state_ppo_v1'
GAMMA = .995
LAMBDA = .95
ACTOR_DIM = 198
CRITIC_DIM = ACTOR_DIM + 26*7 + 5 + 6
ACTION_DIM = 28


def finite(x):
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Nonfinite policy/rollout tensor')
    return x


def mlp(n, out):
    layers = []
    for width in (512, 512, 512):
        layers.extend((nn.Linear(n, width), nn.ELU()))
        n = width
    layers.append(nn.Linear(n, out))
    return nn.Sequential(*layers)


class StateExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = mlp(ACTOR_DIM, ACTION_DIM)
        self.critic = mlp(CRITIC_DIM, 1)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(.5)))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, obs):
        if obs.shape[-1] != ACTOR_DIM:
            raise ValueError('State expert actor observation contract mismatch')
        return torch.distributions.Normal(finite(self.actor(obs)), self.log_std.exp())

    def value(self, critic):
        if critic.shape[-1] != CRITIC_DIM:
            raise ValueError('State expert critic observation contract mismatch')
        return finite(self.critic(critic).squeeze(-1))

    @torch.no_grad()
    def sample(self, obs, critic, deterministic=False):
        d = self.distribution(obs)
        latent = d.mean if deterministic else d.sample()
        # PPO records the sampled latent BEFORE tanh and safety projection.
        # The fixed action transform belongs to the environment; never recompute
        # likelihood from the clipped/projected command.
        return latent, d.log_prob(latent).sum(-1), self.value(critic)


def pack_observation(q, dq, target, core, tip_local, marker_delta, face_deg,
                     progress, reached, remaining, piece_local, contact, core_vel):
    """Simulation-state expert, deliberately NOT described as a vision policy.

    Actor: q56, dq56, target error56, Core pose7, five DIP origins15,
    calibrated-index contact-marker delta3, face angle/progress/retreat/time4,
    normalized target angle1. Critic additionally sees all 26 relative poses,
    five attributed impulse channels and Core velocity.
    """
    n = len(q)
    scalar = torch.stack((face_deg/15, progress, reached.float(), remaining,
                          torch.ones_like(progress)), -1)
    obs = torch.cat((q/math.pi, (dq/5).clamp(-5, 5),
                     ((target-q)/.1).clamp(-5, 5), core,
                     (tip_local/.1).flatten(1), marker_delta/.1, scalar), -1).float()
    priv = piece_local.clone()
    priv[..., :3] /= .1
    critic = torch.cat((obs, priv.flatten(1).float(),
                        torch.log1p(contact.float().clamp_min(0)/.001),
                        core_vel.float()), -1)
    if obs.shape != (n, ACTOR_DIM) or critic.shape != (n, CRITIC_DIM):
        raise ValueError(f'Bad observation shape: {obs.shape}, {critic.shape}')
    return finite(obs), finite(critic)


def potential(distance_m, progress, hold_fraction, retreat_m, reached):
    """Contact-marker shaping + verified object rotation + measured withdrawal.

    Distance is only shaping: it never manufactures physical contact or success.
    Bounded potential differences do not pay repeatedly for sitting in contact.
    """
    contact = 2./(1.+40.*distance_m.clamp_min(0))
    retreat = 2.*(retreat_m/.020).clamp(0, 1) + 2.*hold_fraction.clamp(0, 1)
    return torch.where(reached, 2.+retreat, contact) + 6.*progress.clamp(0, 1)


def step_reward(before, after, done, unsafe, success, latent):
    # Terminal potential is zero, including a finite-horizon timeout. Unsafe
    # terminal transitions get NO positive shaping, contact or success credit.
    shaped = GAMMA*torch.where(done, torch.zeros_like(after), after)-before
    effort = .002*latent.tanh().square().mean(-1)
    reward = shaped - .001 - effort + 20.*success.float()
    reward = torch.where(unsafe, -20.-before.clamp_min(0)-effort, reward)
    return finite(reward)


def gae(reward, value, valid, done, gamma=GAMMA, lam=LAMBDA):
    """Completed finite-horizon episodes only; never cross lanes/prefix gaps.

    Every last valid transition must be terminal. Time budget is part of the
    actor state, so timeout is an MDP terminal, not a bootstrappable truncation.
    """
    if reward.shape != value.shape or valid.shape != value.shape or done.shape != value.shape:
        raise ValueError('Invalid rollout shapes')
    if valid.dtype != torch.bool or done.dtype != torch.bool:
        raise ValueError('Expected boolean rollout masks')
    next_valid = torch.cat((valid[1:], torch.zeros_like(valid[:1])), 0)
    if bool((valid & ~next_valid & ~done).any()):
        raise ValueError('Incomplete episode; refusing zero bootstrap')
    out = torch.zeros_like(reward)
    adv = torch.zeros_like(reward[0]); next_value = adv.clone()
    for t in range(len(reward)-1, -1, -1):
        carry = valid[t] & next_valid[t] & ~done[t]
        delta = reward[t] + gamma*next_value*carry - value[t]
        adv = torch.where(valid[t], delta + gamma*lam*carry*adv, 0.)
        out[t] = adv; next_value = torch.where(valid[t], value[t], 0.)
    return finite(out), finite(out+value)


class PPO:
    def __init__(self, model, lr=3e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, data):
        valid = data['valid']
        if int(valid.sum()) < 32:
            raise ValueError('Fewer than 32 real learned transitions; inspect handoff evidence')
        advantage, returns = gae(data['reward'], data['value'], valid, data['done'])
        a = advantage[valid]
        a = (a-a.mean())/a.std(unbiased=False).clamp_min(1e-6)
        obs, critic = data['obs'][valid], data['critic'][valid]
        latent, old = data['latent'][valid], data['logp'][valid]
        y = returns[valid]
        before = torch.cat([p.detach().flatten().clone() for p in self.model.actor.parameters()])
        history = []; stop = False
        with torch.enable_grad():
            for epoch in range(4):
                for ids in torch.randperm(len(a), device=a.device).split(1024):
                    d = self.model.distribution(obs[ids])
                    logp = d.log_prob(latent[ids]).sum(-1)
                    logratio = logp-old[ids]; ratio = logratio.exp()
                    kl = ((ratio-1)-logratio).mean()
                    if not bool(torch.isfinite(kl)):
                        raise ValueError('Nonfinite PPO KL')
                    if float(kl.detach()) > .03:
                        stop = True; break
                    policy_loss = -torch.minimum(ratio*a[ids], ratio.clamp(.8, 1.2)*a[ids]).mean()
                    value_loss = (self.model.value(critic[ids])-y[ids]).square().mean()
                    loss = finite(policy_loss + .5*value_loss - .001*d.entropy().sum(-1).mean())
                    self.optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
                    self.optimizer.step()
                    with torch.no_grad(): self.model.log_std.clamp_(math.log(.08), math.log(.8))
                    history.append((float(policy_loss.detach()), float(value_loss.detach()), float(kl.detach()), float(norm)))
                if stop: break
        if not history:
            raise ValueError('PPO made no update; stale/incompatible on-policy data')
        change = float((torch.cat([p.detach().flatten() for p in self.model.actor.parameters()])-before).abs().max())
        if not math.isfinite(change) or change <= 0:
            raise ValueError('Actor parameters did not change')
        return dict(real_transitions=int(valid.sum()), optimizer_steps=len(history),
                    actor_max_delta=change, mean_reward=float(data['reward'][valid].mean()),
                    policy_loss=history[-1][0], value_loss=history[-1][1],
                    approx_kl=history[-1][2], kl_early_stop=stop)
