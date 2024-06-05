import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path

import utils
from dm_control.utils import rewards
from agent_example import Agent, load_data


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=1e-3):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())

        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        f = self.policy(obs)
        mu = self.fc_mu(f)
        log_std = self.fc_logstd(f)

        mu = torch.clamp(mu, -5, 5)

        std = log_std.clamp(-5, 2).exp()
        dist = utils.SquashedNormal2(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=1e-3):
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.q1_last = nn.Linear(hidden_dim, 1)

        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.q2_last = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q1 = self.q1_last(q1)

        q2 = self.q2_net(obs_action)
        q2 = self.q2_last(q2)

        return q1, q2
    

class CDSDiscriminator(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.state_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
        self.state_action_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        s_logits = self.state_net(obs)
        sa_logits = self.state_action_net(obs_action)
        return s_logits, sa_logits


class DWCDSAgent(Agent):
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 actor_lr,
                 critic_lr,
                 hidden_dim,
                 critic_target_tau,
                 clip_grad_norm,
                 nstep,
                 batch_size,
                 use_tb,
                 alpha,
                 n_samples,
                 target_cql_penalty,
                 discriminator_clip_ratio,
                 discriminator_lr,
                 flow_discount,
                 discriminator_weight_temp,
                 flow_weight,
                 kl_weight,
                 use_critic_lagrange,
                 num_expl_steps,
                 has_next_action=False):

        self.num_expl_steps = num_expl_steps
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.clip_grad_norm = clip_grad_norm
        self.use_tb = use_tb
        self.use_critic_lagrange = use_critic_lagrange
        self.target_cql_penalty = target_cql_penalty
        self.discriminator_clip_ratio = discriminator_clip_ratio
        self.flow_discount = flow_discount
        self.discriminator_weight_temp = discriminator_weight_temp
        self.flow_weight = flow_weight
        self.kl_weight = kl_weight

        self.alpha = alpha
        self.n_samples = n_samples

        state_dim = obs_shape[0]
        action_dim = action_shape[0]

        # models
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.discriminator = CDSDiscriminator(state_dim, action_dim, hidden_dim).to(device)

        # lagrange multipliers
        self.target_entropy = -self.action_dim
        self.log_actor_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.log_critic_alpha = torch.zeros(1, requires_grad=True, device=device)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_alpha_opt = torch.optim.Adam([self.log_actor_alpha], lr=actor_lr)
        self.critic_alpha_opt = torch.optim.Adam([self.log_critic_alpha], lr=actor_lr)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)

        self.train()
        # self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        policy = self.actor(obs)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().detach().numpy()[0]

    def _repeated_critic_apply(self, obs, actions):
        """
            obs is (batch_size, obs_dim)
            actions is (n_samples, batch_size, action_dim)

            output tensors are (n_samples, batch_size, 1)
        """
        batch_size = obs.shape[0]
        n_samples = actions.shape[0]

        reshaped_actions = actions.reshape((n_samples * batch_size, -1))
        repeated_obs = obs.unsqueeze(0).repeat((n_samples, 1, 1))
        repeated_obs = repeated_obs.reshape((n_samples * batch_size, -1))

        Q1, Q2 = self.critic(repeated_obs, reshaped_actions)
        Q1 = Q1.reshape((n_samples, batch_size, 1))
        Q2 = Q2.reshape((n_samples, batch_size, 1))

        return Q1, Q2

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Compute standard SAC loss
        with torch.no_grad():
            dist = self.actor(next_obs)                 # SquashedNormal分布
            sampled_next_action = dist.sample()         # (1024, act_dim)
            next_action_log_prob = dist.log_prob(sampled_next_action) # (1024, )
            # print("sampled_next_action:", sampled_next_action.shape)
            target_Q1, target_Q2 = self.critic_target(next_obs, sampled_next_action)  # (1024,1), (1024,1)
            target_V = torch.min(target_Q1, target_Q2)  # (1024,1)
            alpha = self.log_actor_alpha.exp().detach()
            target_V = target_V - alpha * next_action_log_prob.unsqueeze(-1)  # (1024,1)
            target_Q = reward + (discount * target_V)   # (1024,1)

        Q1, Q2 = self.critic(obs, action)
        # Q1_weighted = Q1 * weights.unsqueeze(-1)
        # Q2_weighted = Q2 * weights.unsqueeze(-1)
        # target_Q_weighted = target_Q * weights.unsqueeze(-1)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)   # 标量

        # Add CQL penalty
        with torch.no_grad():
            random_actions = torch.FloatTensor(self.n_samples, Q1.shape[0],
                        action.shape[-1]).uniform_(-1, 1).to(self.device)    # (n_samples, 1024, act_dim)
            sampled_actions = self.actor(obs).sample(
                sample_shape=(self.n_samples,))                              # (n_samples, 1024, act_dim)
            # print("sampled_actions:", sampled_actions.shape)
            next_sampled_actions = self.actor(next_obs).sample(
                sample_shape=(self.n_samples,))                              # (n_samples, 1024, act_dim)
            # print("next_sampled_actions:", next_sampled_actions.shape)

        rand_Q1, rand_Q2 = self._repeated_critic_apply(obs, random_actions)  # (n_samples, 1024, 1)
        sampled_Q1, sampled_Q2 = self._repeated_critic_apply(                # (n_samples, 1024, 1)
            obs, sampled_actions)
        next_sampled_Q1, next_sampled_Q2 = self._repeated_critic_apply(      # (n_samples, 1024, 1)
            obs, next_sampled_actions)

        # situation 1
        cat_Q1 = torch.cat([rand_Q1, sampled_Q1, next_sampled_Q1,
                            Q1.unsqueeze(0)], dim=0)                # (1+3*n_samples, 1024, 1)
        cat_Q2 = torch.cat([rand_Q2, sampled_Q2, next_sampled_Q2,
                            Q2.unsqueeze(0)], dim=0)                # (1+3*n_samples, 1024, 1)

        assert (not torch.isnan(cat_Q1).any()) and (not torch.isnan(cat_Q2).any())

        cql_logsumexp1 = torch.mean(torch.logsumexp(cat_Q1, dim=0)) # (1024, 1)
        cql_logsumexp2 = torch.mean(torch.logsumexp(cat_Q2, dim=0)) # (1024, 1)
        cql_logsumexp = cql_logsumexp1 + cql_logsumexp2

        cql_penalty = cql_logsumexp - (Q1 + Q2).mean()  # 标量

        # Update lagrange multiplier
        if self.use_critic_lagrange:
            alpha = torch.clamp(self.log_critic_alpha.exp(), min=0.0, max=1000000.0)
            alpha_loss = -0.5 * alpha * (cql_penalty - self.target_cql_penalty)

            self.critic_alpha_opt.zero_grad()
            alpha_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_([self.log_critic_alpha], self.clip_grad_norm)
            self.critic_alpha_opt.step()
            alpha = torch.clamp(self.log_critic_alpha.exp(),
                                min=0.0,
                                max=1000000.0).detach()
        else:
            alpha = self.alpha

        # Combine losses
        critic_loss = critic_loss + alpha * cql_penalty

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = utils.grad_norm(self.critic.parameters())
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        self.critic_opt.step()

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            # metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['critic_cql'] = cql_penalty.item()
            metrics['critic_cql_logsum1'] = cql_logsumexp1.item()
            # metrics['critic_cql_logsum2'] = cql_logsumexp1.item()
            metrics['rand_Q1'] = rand_Q1.mean().item()
            # metrics['rand_Q2'] = rand_Q2.mean().item()
            metrics['sampled_Q1'] = sampled_Q1.mean().item()
            # metrics['sampled_Q2'] = sampled_Q2.mean().item()
            metrics['critic_grad_norm'] = critic_grad_norm
            metrics['critic_alpha'] = alpha

        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        policy = self.actor(obs)
        sampled_action = policy.rsample()              # (1024, 6)
        # print("sampled_action:", sampled_action.shape)
        log_pi = policy.log_prob(sampled_action)       # (1024, )

        # update lagrange multiplier
        alpha_loss = -(self.log_actor_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        self.actor_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        nn.utils.clip_grad_norm_([self.log_actor_alpha], self.clip_grad_norm)
        self.actor_alpha_opt.step()
        alpha = self.log_actor_alpha.exp().detach()

        # optimize actor
        Q1, Q2 = self.critic(obs, sampled_action)     # (1024, 1)
        Q = torch.min(Q1, Q2)                         # (1024, 1)
        actor_loss = (alpha * log_pi - torch.squeeze(Q)).mean()      # 标量
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        actor_grad_norm = utils.grad_norm(self.actor.parameters())
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = -log_pi.mean().item()
            metrics['actor_alpha'] = alpha.item()
            metrics['actor_alpha_loss'] = alpha_loss.item()
            metrics['actor_mean'] = policy.loc.mean().item()
            metrics['actor_std'] = policy.scale.mean().item()
            metrics['actor_action'] = sampled_action.mean().item()
            metrics['actor_atanh_action'] = utils.atanh(sampled_action).mean().item()
            metrics['actor_grad_norm'] = actor_grad_norm

        return metrics
    
    def update_discriminator(self, reward, normal_logits, s_logits, sa_logits, next_s_logits, step):
        metrics = dict()
        
        normal_reward = torch.squeeze((reward - torch.mean(reward)) / (torch.std(reward) + 1e-6), dim=-1)
        policy_loss = -torch.sum(normal_reward * normal_logits)
        consistency_loss = F.mse_loss(self.flow_discount * torch.exp((s_logits + sa_logits) / self.discriminator_weight_temp), torch.exp(next_s_logits / self.discriminator_weight_temp))
        kl_loss = torch.sum(normal_logits * torch.log(normal_logits))
        discriminator_loss = policy_loss + self.flow_weight * consistency_loss + self.kl_weight * kl_loss
        
        self.discriminator_opt.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        discriminator_grad_norm = utils.grad_norm(self.discriminator.parameters())
        nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.clip_grad_norm)
        self.discriminator_opt.step()
        
        if self.use_tb:
            metrics['discriminator_loss'] = discriminator_loss.item()
            metrics['policy_loss'] = policy_loss.item()
            metrics['consistency_loss'] = consistency_loss.item()
            metrics['kl_loss'] = kl_loss.item()
            metrics['discriminator_grad_norm'] = discriminator_grad_norm
            
        return metrics


    def conservative_data_share(self, batch_main, batch_share):
        obs_all, action_all, next_obs_all, reward_all, discount_all, obs_k_all = [], [], [], [], [], []

        # 1. sample examples from the main task
        # shape = (512, 24), (512, 6), (512, 1), (512, 1), (512, 24)
        obs_m, action_m, reward_m, discount_m, next_obs_m, _ = utils.to_torch(batch_main, self.device)
        # print("main samples:", obs_m.shape, action_m.shape, reward_m.shape, discount_m.shape, next_obs_m.shape, eps_flag_m.shape)
        obs_all.append(obs_m)
        action_all.append(action_m)
        next_obs_all.append(next_obs_m)
        reward_all.append(reward_m)
        discount_all.append(discount_m)

        # 2. sample examples from other tasks
        # sample 10 times samples, and select the top-0.1 samples.   (batch_size_split*10, xx, xx)
        # shape = (5120, 24) (5120, 6) (5120, 1) (5120, 1) (5120, 24)
        obs, action, reward, discount, next_obs, _ = utils.to_torch(batch_share, self.device)
        # print("obs:", obs.shape, action.shape, reward.shape, discount.shape, next_obs.shape)
        # calculate the conservative Q value
        with torch.no_grad():
            conservative_q_value, _ = self.critic(obs, action)          # (5120, 1)
            # main_q_value, _ = self.critic(obs_m, action_m)              # (512, 1)
            share_s_logits, share_sa_logits = self.discriminator(obs, action)
            # main_s_logits, main_sa_logits = self.discriminator(obs_m, action_m)
            clipped_s_logits = (share_s_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
            clipped_sa_logits = (share_sa_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
            # clipped_main_s_logits = (main_s_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
            # clipped_main_sa_logits = (main_sa_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
            # all_sa_logits = torch.cat([clipped_s_logits + clipped_sa_logits, clipped_main_s_logits + clipped_main_sa_logits], dim=0) # (5632, 1)
            normal_logits = F.softmax(torch.squeeze(clipped_s_logits + clipped_sa_logits, dim=-1), dim=0) # (5632, )
            conservative_q_value = normal_logits.unsqueeze(-1) * conservative_q_value
            # main_q_value = normal_logits[-main_q_value.shape[0]:].unsqueeze(-1) * main_q_value
            
        conservative_q_value = conservative_q_value.squeeze().detach().cpu().numpy()
        # main_q_value = main_q_value.squeeze().detach().cpu().numpy()
        # choose the top 0.1  top_index.shape=(512,)
        # main_pivot = np.argpartition(main_q_value, -int(0.5 * obs_m.shape[0]))[-int(0.5 * obs_m.shape[0]):]  # find the top 0.1 index. (batch_size_split,)
        top_index = np.argpartition(conservative_q_value, -obs_m.shape[0])[-obs_m.shape[0]:]
        reweight_s, reweight_sa = self.discriminator(obs[top_index], action[top_index])
        clipped_reweight_s = (reweight_s / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
        clipped_reweight_sa = (reweight_sa / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
        reweight = torch.exp(clipped_reweight_s + clipped_reweight_sa)  # (512, 1)
        
        # extract the samples
        obs_all.append(obs[top_index])
        action_all.append(action[top_index])
        next_obs_all.append(next_obs[top_index])
        reward_all.append(reward[top_index] * reweight)
        discount_all.append(discount[top_index])

        return torch.cat(obs_all, dim=0), torch.cat(action_all, dim=0), torch.cat(reward_all, dim=0), \
            torch.cat(discount_all, dim=0), torch.cat(next_obs_all, dim=0)

    def update(self, replay_iter_main, replay_iter_share, step, total_step):
        metrics = dict()

        batch_main = next(replay_iter_main)  # len=6, inner_shape=512x24
        batch_share = next(replay_iter_share)
        
        obs_m, action_m, reward_m, discount_m, next_obs_m, _ = utils.to_torch(batch_main, self.device)
        
        if step % 5 == 0:
            # Predict weights
            s_logits, sa_logits = self.discriminator(obs_m, action_m)
            next_s_logits, _ = self.discriminator(next_obs_m, action_m)
            clipped_s_logits = (s_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
            clipped_sa_logits = (sa_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
            normal_logits = F.softmax(torch.squeeze(clipped_s_logits + clipped_sa_logits, dim=-1), dim=0) # (1024, )
            
            # update discriminator
            metrics.update(self.update_discriminator(reward_m, normal_logits, s_logits, sa_logits, next_s_logits, step))
        

        # print("conservative data sharing...")   # obs.shape=(1024, 24), action.shape=(1024, 6) reward.shape=(1024, 1)
        obs, action, reward, discount, next_obs = self.conservative_data_share(batch_main, batch_share)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
            
        # Predict weights
        s_logits, sa_logits = self.discriminator(obs, action)
        next_s_logits, _ = self.discriminator(next_obs, action)
        clipped_s_logits = (s_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
        clipped_sa_logits = (sa_logits / self.discriminator_weight_temp).clamp(-1. - self.discriminator_clip_ratio, 1. + self.discriminator_clip_ratio)
        normal_logits = F.softmax(torch.squeeze(clipped_s_logits + clipped_sa_logits, dim=-1), dim=0) # (1024, )
        # weights = normal_logits.detach() # (1024, )

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def save(self, dir: Path) -> None:
        print("Save new best model")
        torch.save(self.actor, dir / "actor.pth")
        torch.save(self.critic, dir / "critic.pth")
        torch.save(self.discriminator, dir / "discriminator.pth")
    
    def load(self, load_path: Path) -> None:
        self.actor = torch.load(load_path / "actor.pth")
        self.critic = torch.load(load_path / "critic.pth")
        self.discriminator = torch.load(load_path / "discriminator.pth")