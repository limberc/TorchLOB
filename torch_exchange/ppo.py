import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from typing import Tuple, List
from tqdm import tqdm
from .models.networks import ActorCritic

# --- Constants & Config ---
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_KEY = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


class PPOAgent:
    def __init__(self, env, device='cpu', 
                 lr=3e-4, 
                 model_type='mlp',
                 book_depth=10,
                 n_stack=4,
                 safety_mode='none',  
                 # 'none', 'rcpo', 'ipo', 'pid'
                 cost_limit=0.01,
                 lagrange_lr=0.01,
                 pid_Kp=0.01, pid_Ki=0.001, pid_Kd=0.0):
        self.env = env
        self.device = device
        self.model_type = model_type
        self.book_depth = book_depth
        self.n_stack = n_stack
        self.safety_mode = safety_mode
        self.cost_limit = cost_limit
        
        # Dimensions
        raw_obs_dim = env.observation_space.shape[0]
        
        if model_type == 'cnn':
            # Input to network will be stacked
            obs_dim = raw_obs_dim * n_stack
        else:
            obs_dim = raw_obs_dim
            
        action_dim = env.action_space.shape[0]
        
        self.network = ActorCritic(obs_dim, action_dim, model_type, book_depth, n_stack).to(device)
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
        # Prepare initial observation once
        obs, _ = self.env.reset()
        
        # Frame Stacking State
        obs_stack = None
        if self.model_type == 'cnn':
            # Stack shape: (n_stack, raw_obs_dim)
            obs_stack = obs.unsqueeze(0).repeat(self.n_stack, 1)
        
        # Helper to get network input
        def get_net_input(o_stack):
             if self.model_type == 'cnn':
                 return o_stack.view(1, -1) # Flatten stack to (1, n_stack*obs)
             else:
                 return o_stack.unsqueeze(0) # (1, obs)

        if self.model_type != 'cnn':
             obs = obs # Just consistent naming
        
        while global_step < total_timesteps:
            # 1. Collect Rollout            
            batch_rewards = []
            batch_costs = []
            
            for i in range(batch_size):
                
                # Prepare input
                if self.model_type == 'cnn':
                    net_input = get_net_input(obs_stack)
                else:
                    net_input = get_net_input(obs)
                
                with torch.no_grad():
                    action, logprob, _, value = self.network.get_action_and_value(net_input)
                
                action_exec = torch.abs(action).int().flatten()
                
                next_obs, reward, terminated, truncated, info = self.env.step(action_exec)
                done = terminated or truncated
                
                cost = info.get('cost', 0.0)
                
                # Metrics
                if isinstance(reward, torch.Tensor): r_val = reward.item()
                else: r_val = reward
                
                if isinstance(cost, torch.Tensor): c_val = cost.item()
                else: c_val = cost
                
                batch_rewards.append(r_val)
                batch_costs.append(c_val)
                
                # Store
                if self.model_type == 'cnn':
                     obs_buf.append(net_input.squeeze(0)) # Store flattened stack
                else:
                     obs_buf.append(obs)
                     
                actions_buf.append(action)
                logprobs_buf.append(logprob)
                rewards_buf.append(r_val) 
                costs_buf.append(c_val)
                dones_buf.append(done)
                values_buf.append(value)
                
                # State Update
                if self.model_type == 'cnn':
                    # Shift stack
                    obs_stack = torch.cat((obs_stack[1:], next_obs.unsqueeze(0)), dim=0)
                else:
                    obs = next_obs
                
                global_step += 1
                pbar.update(1)
                
                if (i + 1) % 100 == 0:
                     current_avg_reward = np.mean(batch_rewards[-100:])
                     current_avg_cost = np.mean(batch_costs[-100:])
                     pbar.set_postfix({
                        'reward (roll)': f"{current_avg_reward:.2f}",
                        'cost (roll)': f"{current_avg_cost:.2f}",
                        'loss': f"{last_loss:.3f}" if last_loss != 0 else "collecting..."
                    })
                
                if done:
                    # Next Value bootstrapping needs correct next_obs
                    # But we are done, so next value is masked anyway.
                    # Just define correct bootstrapping state if needed?
                    # PPO uses next_value for advantage calc at end of batch. 
                    # If this step is done, nextnonterminal is 0.
                    
                    # Reset Env
                    reset_obs, _ = self.env.reset()
                    
                    if self.model_type == 'cnn':
                        obs_stack = reset_obs.unsqueeze(0).repeat(self.n_stack, 1)
                    else:
                        obs = reset_obs

            # 2. Compute Advantages
            with torch.no_grad():
                # Get next value
                if self.model_type == 'cnn':
                     # We need next state stack. 
                     # If done at last step, obs_stack is already reset. This is tricky for GAE.
                     # GAE needs V(s_{t+1}).
                     # If last step was done, next_value is V(s_{initial}), but masked by done=1.
                     # So it doesn't matter what V is.
                     # If NOT done, obs_stack is valid s_{t+1}.
                     next_net_input = get_net_input(obs_stack)
                else:
                     next_net_input = get_net_input(obs)
                     
                _, _, _, next_value = self.network.get_action_and_value(next_net_input)
                
            advantages = torch.zeros(batch_size, device=self.device)
            lastgaelam = 0
            
            # Convert buffers to tensors
            b_obs = torch.stack(obs_buf) # stack list of tensors
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
