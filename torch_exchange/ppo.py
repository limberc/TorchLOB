import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from typing import Tuple, List
from tqdm import tqdm

# --- Constants & Config ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_KEY = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
HIDDEN_DIM = 256

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh()
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(HIDDEN_DIM, action_dim),
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 1)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        return features
        
    def get_action_and_value(self, x, action=None):
        features = self.forward(x)
        
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        action_log_probs = probs.log_prob(action).sum(axis=-1)
        dist_entropy = probs.entropy().sum(axis=-1)
        value = self.critic(features)
        
        return action, action_log_probs, dist_entropy, value

class PPOAgent:
    def __init__(self, env, device='cpu', 
                 lr=3e-4, 
                 safety_mode='none',  
                 # 'none', 'rcpo', 'ipo', 'pid'
                 cost_limit=0.01,
                 lagrange_lr=0.01,
                 pid_Kp=0.01, pid_Ki=0.001, pid_Kd=0.0):
        self.env = env
        self.device = device
        self.safety_mode = safety_mode
        self.cost_limit = cost_limit
        
        # Dimensions
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        self.network = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # --- Safety Params ---
        # Lagrange Multiplier (Lambda)
        self.log_lambda = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.lambda_optimizer = optim.Adam([self.log_lambda], lr=lagrange_lr)
        
        # IPO
        self.ipo_t = 1000.0 # Multiplier for Log Barrier
        
        # PID
        self.pid_Kp = pid_Kp
        self.pid_Ki = pid_Ki
        self.pid_Kd = pid_Kd
        self.pid_I = 0.0
        self.pid_prev_error = 0.0
        self.lambda_val = 0.0 # Explicit value for PID mode

    def train(self, total_timesteps=100000, batch_size=2048, mini_batch_size=64, epochs=10):
        obs_buf = []
        actions_buf = []
        logprobs_buf = []
        rewards_buf = []
        costs_buf = [] # Constraint Signal
        dones_buf = []
        values_buf = []
        
        global_step = 0
        obs = self.env.reset()
        
        pbar = tqdm(total=total_timesteps, desc="Training")
        
        # Metrics trackers
        avg_reward = 0.0
        avg_cost = 0.0
        last_loss = 0.0
        
        # Prepare initial observation once
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        while global_step < total_timesteps:
            # 1. Collect Rollout            
            # Temporary buffers for calculating simple averages for the progress bar during collection
            batch_rewards = []
            batch_costs = []
            
            for i in range(batch_size):
                with torch.no_grad():
                    action, logprob, _, value = self.network.get_action_and_value(obs)
                
                action_np = action.cpu().numpy().flatten()
                action_exec = np.abs(action_np).astype(np.int32)
                
                next_obs, reward, done, info = self.env.step(action_exec)
                
                cost = info.get('cost', 0.0)
                
                # Update temporary metrics
                if isinstance(reward, torch.Tensor): reward = reward.item()
                if isinstance(cost, torch.Tensor): cost = cost.item()
                
                batch_rewards.append(reward)
                batch_costs.append(cost)
                
                # Store raw reward (normalization happens later or via running stat)
                obs_buf.append(obs)
                actions_buf.append(action)
                logprobs_buf.append(logprob)
                rewards_buf.append(reward) 
                costs_buf.append(cost)
                dones_buf.append(done)
                values_buf.append(value)
                
                obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                global_step += 1
                pbar.update(1)
                
                # Update Progress Bar metrics every 100 steps during collection
                if (i + 1) % 100 == 0:
                     current_avg_reward = np.mean(batch_rewards[-100:])
                     current_avg_cost = np.mean(batch_costs[-100:])
                     pbar.set_postfix({
                        'reward (roll)': f"{current_avg_reward:.2f}",
                        'cost (roll)': f"{current_avg_cost:.2f}",
                        'loss': f"{last_loss:.3f}" if last_loss != 0 else "collecting..."
                    })
                
                if done:
                    obs = self.env.reset()
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 2. Compute Advantages
            with torch.no_grad():
                _, _, _, next_value = self.network.get_action_and_value(obs)
                
            advantages = torch.zeros(batch_size, device=self.device)
            lastgaelam = 0
            


            # Convert buffers to tensors
            b_obs = torch.cat(obs_buf)
            b_actions = torch.cat(actions_buf)
            b_logprobs = torch.cat(logprobs_buf)
            b_rewards = torch.tensor(rewards_buf, device=self.device, dtype=torch.float32)
            b_costs = torch.tensor(costs_buf, device=self.device, dtype=torch.float32)
            b_dones = torch.tensor(dones_buf, device=self.device).float()
            b_values = torch.cat(values_buf).squeeze()
            
            # --- Reward Normalization ---
            # Now handled fundamentally by the Environment (Scaling by Initial Value).
            # b_rewards are already in range ~[-1.0, 1.0] or similar.
            # No extra processing needed here.
            
            for t in reversed(range(batch_size)):
                if t == batch_size - 1:
                    nextnonterminal = 1.0
                    nextvalues = next_value.squeeze()
                else:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = b_values[t+1]
                    
                delta = b_rewards[t] + GAMMA * nextvalues * nextnonterminal - b_values[t]
                lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
                
            returns = advantages + b_values
            b_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # --- Safety Update (PID / RCPO Lambda Update) ---
            mean_cost = b_costs.mean()
            
            if self.safety_mode == 'pid':
                error = mean_cost.item() - self.cost_limit
                self.pid_I += error
                D = error - self.pid_prev_error
                self.lambda_val = max(0, self.pid_Kp * error + self.pid_Ki * self.pid_I + self.pid_Kd * D)
                self.pid_prev_error = error
                
            elif self.safety_mode == 'rcpo':
                # Gradient Ascent on Lambda: lambda * (cost - limit)
                # We want to maximize this w.r.t lambda (Dual Ascent)
                # Loss for lambda optimizer = - (lambda * (cost - limit))
                # Softplus lambda for positivity
                lambda_param = nn.functional.softplus(self.log_lambda)
                lambda_loss = -lambda_param * (mean_cost - self.cost_limit)
                
                self.lambda_optimizer.zero_grad()
                lambda_loss.backward()
                self.lambda_optimizer.step()
                
            
            # 3. PPO Update
            inds = np.arange(batch_size)
            
            for epoch in range(epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, mini_batch_size):
                    end = start + mini_batch_size
                    mb_inds = inds[start:end]
                    
                    _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        
                    mb_advantages = b_advantages[mb_inds]
                    
                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value Loss
                    newvalue = newvalue.squeeze()
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()
                    
                    # Safety Penalty
                    penalty = 0.0
                    current_lambda = 0.0
                    
                    if self.safety_mode == 'rcpo':
                        current_lambda = nn.functional.softplus(self.log_lambda).detach()
                        # Penalty = lambda * cost
                         # Note: cost is per-batch-item. using mean for simplicity or per item?
                        # Often applied as reward penalty: Reward -= lambda * cost.
                        # Here added to Loss: + lambda * cost_term
                        # We use mean cost of minibatch
                        penalty = current_lambda * (b_costs[mb_inds].mean() - self.cost_limit)
                        
                    elif self.safety_mode == 'ipo':
                        # Log Barrier: - t * log(limit - cost)
                        # Applied to objective. Loss = - Objective.
                        # Loss += t * log(Limit - Cost) ??
                        # Actually IPO constraint: Mean Cost <= Limit
                        # Log Barrier: log(Limit - Cost)
                        # We maximize Objective + t * log(Limit - Cost)
                        # Loss = -Objective - t * log(Limit - Cost)
                        mb_mean_cost = b_costs[mb_inds].mean()
                        if mb_mean_cost < self.cost_limit:
                            penalty = -self.ipo_t * torch.log(self.cost_limit - mb_mean_cost)
                        else:
                            penalty = 1000.0 # High penalty if violated
                            
                    elif self.safety_mode == 'pid':
                        # Fixed Lambda from PID controller
                        penalty = self.lambda_val * (b_costs[mb_inds].mean() - self.cost_limit)
                    
                    
                    # Total Loss
                    loss = pg_loss - ENT_KEY * entropy.mean() + VF_COEF * v_loss + penalty
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                    self.optimizer.step()
            
            # Post-batch update
            last_loss = v_loss.item()
            avg_reward = b_rewards.mean().item()
            avg_cost = b_costs.mean().item()
            
            # Logging
            lambda_log = 0.0
            if self.safety_mode == 'rcpo':
                 lambda_log = nn.functional.softplus(self.log_lambda).item()
            elif self.safety_mode == 'pid':
                 lambda_log = self.lambda_val
            
            pbar.set_postfix({
                'batch_rew': f"{avg_reward:.2f}",
                'batch_cost': f"{avg_cost:.2f}",
                'loss': f"{last_loss:.3f}",
                'lambda': f"{lambda_log:.4f}"
            })
            # print(f"Global Step: {global_step} | Mean Reward: {b_rewards.mean().item():.4f} | Mean Cost (Slippage): {b_costs.mean().item():.4f} | Lambda: {lambda_log:.4f}")
            
            # Clear buffers
            obs_buf = []
            actions_buf = []
            logprobs_buf = []
            rewards_buf = []
            costs_buf = []
            dones_buf = []
            values_buf = []
        
        pbar.close()
