import torch
import gym
from gym import spaces
import numpy as np
from typing import Dict, Tuple, Optional
from torch_exchange.orderbook import (
    init_orderside, add_order, cancel_order, match_order, 
    get_best_bid_and_ask_inclQuants, get_L2_state,
    get_best_bid_idx, get_best_ask_idx
)

# Constants
MAX_INT = 2_147_483_647

class TorchExecutionEnv(gym.Env):
    """
    A PyTorch-based Limit Order Book (LOB) Execution Environment.
    
    Simulates a market environment where an agent must execute a trade (buy/sell) 
    of a specific quantity (`task_size`) within a fixed time window.
    
    Parameters:
    - task (str): 'sell' or 'buy'. Direction of the trade.
    - task_size (int): Total number of units to execute (e.g., 5000 shares). 
      The agent starts with this inventory and must reduce it to 0.
    - book_depth (int): Number of price levels to observe on each side (Bid/Ask). 
      Example: If 10, the agent sees the best 10 Bids and 10 Asks.
    - n_actions (int, implied): The action space size, derived from `book_depth`. 
      The agent can choose to place orders at different price levels.
      Typically matches or relates to `book_depth`.
    - slice_time_window (int): Max time steps allowed for the episode.
    - tick_size (int, implied): The minimum price increment for orders (default 1).
      Prices must be multiples of this value.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, task='sell', task_size=5000, device='cpu',
    slice_time_window=1800, book_depth=10, tick_size=1):
        super(TorchExecutionEnv, self).__init__()
        self.device = device
        self.task = task
        self.task_size = task_size
        self.n_actions = book_depth
        self.tick_size = tick_size
        self.n_ticks_in_book = 2
        
        # Action space: 0-3 corresponding to [FT, M, NT, PP]
        # Using discrete actions for simplicity matching the simpler demo logic
        # OR defining continuous/multi-discrete as per original exec_env?
        # Original exec_env had Box(-5,5) or Box(0,100).
        # We'll use Box for quantities, but simplify for demo.
        # Let's stick to the structure: Action is a dictionary or array?
        # original step takes 'action' which is a dict or array. 
        # In demo notebook loop: action = env.action_space(env_params).sample(key_act)
        # We will expose a Box action space representing Quantities for the 4 price levels.
        self.action_space = spaces.Box(low=0, high=10, shape=(self.n_actions,), dtype=np.int32)
        
        # Observation space: 610 dim vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(610,), dtype=np.float32)

        # Internal State placeholders
        self.state = None
        
        # Mock Data Parameters
        self.sliceTimeWindow = slice_time_window
        self.stepLines = 100
        self.nOrdersPerSide = 100
        self.nTradesLogged = 100
        self.book_depth = book_depth
        self.trader_unique_id = -9000 + 1
        
        # Initial Price for Mock Data
        self.init_price = 1000
        self.arrival_mid_price = self.init_price

    def reset(self):
        # Initialize Order Book State
        self.ask_orders = init_orderside(self.nOrdersPerSide, self.device)
        self.bid_orders = init_orderside(self.nOrdersPerSide, self.device)
        self.trades = (torch.ones((self.nTradesLogged, 6), dtype=torch.int32, device=self.device) * -1)
        
        # --- Populate Mock LOB to have some initial liquidity ---
        # Asks: 100000, 100100, ...
        # Bids: 99900, 99800, ...
        for i in range(10):
            # Add Ask
            msg_ask = {
                'price': self.init_price + (i + 1) * 100,
                'quantity': 100,
                'orderid': i + 1000, # Dummy ID
                'traderid': 0,
                'time': 0,
                'time_ns': 0
            }
            self.ask_orders = add_order(self.ask_orders, msg_ask)
            
            # Add Bid
            msg_bid = {
                'price': self.init_price - (i + 1) * 100,
                'quantity': 100,
                'orderid': i + 2000,
                'traderid': 0,
                'time': 0,
                'time_ns': 0
            }
            self.bid_orders = add_order(self.bid_orders, msg_bid)
        
        # State tracking
        self.quant_executed = 0
        self.task_to_execute = self.task_size
        self.step_counter = 0
        self.max_steps = 100
        self.total_revenue = 0
        self.time = 0
        
        # Track Arrival Mid Price for Slippage Calculation
        best_ask, best_bid = get_best_bid_and_ask_inclQuants(self.ask_orders, self.bid_orders)
        if best_ask[0] != MAX_INT and best_bid[0] != -1:
             self.arrival_mid_price = (best_ask[0] + best_bid[0]) / 2.0
        else:
             self.arrival_mid_price = self.init_price

        return self._get_obs()

    def step(self, action):
        """
        action: array of shape (4,) representing quantities to submit at [FT, M, NT, PP] prices.
        """
        # 1. Determine Prices
        # Get BBO
        best_ask, best_bid = get_best_bid_and_ask_inclQuants(self.ask_orders, self.bid_orders)
        best_ask_price = best_ask[0].item()
        best_bid_price = best_bid[0].item()
        
        if best_bid_price == -1: best_bid_price = self.init_price - 100
        if best_ask_price == MAX_INT: best_ask_price = self.init_price + 100

        # Define Price Levels logic from exec_env.py
        # FT = Far Touch
        # M = Mid
        # NT = Near Touch
        # PP = Passive
        
        if self.task == 'sell':
            # Sell prices are typically Ask side or matching Bids
            # exec_env logic:
            # FT = best_bid
            # M = (best_bid + best_ask) // 2
            # NT = best_ask
            # PP = best_ask + ticks
            
            # Determine target price levels based on n_actions
            # Previous Hardcoded: [FT, M, NT, PP] (4 levels)
            # Dynamic: Generate n_actions levels.
            # Strategy:
            # Level 0: Far Touch (Aggressive) -> Crossing the spread (Best Bid for Sell)
            # Level 1: Mid Price
            # Level 2...N: Near Touch (Passive) + increments (Best Ask + k*tick for Sell)
            
            price_levels = []
            
            # --- SELL TASK ---
            if self.task == 'sell':
                # Aggressive (execute against bids)
                price_levels.append(best_bid_price) 
                
                # Mid
                mid = (best_bid_price + best_ask_price) // 2 // self.tick_size * self.tick_size
                price_levels.append(mid)
                
                # Passive sequence starting from Best Ask
                # Remaining levels: n_actions - 2
                start_passive = best_ask_price
                for i in range(self.n_actions - 2):
                    p = start_passive + i * self.tick_size
                    price_levels.append(p)
                    
            # --- BUY TASK ---
            else: 
                # Aggressive (execute against asks)
                price_levels.append(best_ask_price)
                
                # Mid
                mid = (best_bid_price + best_ask_price) // 2 // self.tick_size * self.tick_size
                price_levels.append(mid)
                
                # Passive sequence starting from Best Bid (going down)
                start_passive = best_bid_price
                for i in range(self.n_actions - 2):
                     p = start_passive - i * self.tick_size
                     price_levels.append(p)
                     
            # Safety fill if n_actions < 2 (unlikely but safe)
            while len(price_levels) < self.n_actions:
                price_levels.append(price_levels[-1])
            
            # Truncate if somehow larger (logic above prevents this, but safe practice)
            price_levels = price_levels[:self.n_actions]

        # 2. Convert Action to Orders
        # action is quantities
        try:
            quants = torch.tensor(action, dtype=torch.int32, device=self.device)
        except:
             quants = torch.tensor(action, dtype=torch.int32, device=self.device)
             
        # Clip to task size
        remaining_task = self.task_to_execute - self.quant_executed
        # Basic trimming logic (simplification)
        total_q = torch.sum(quants)
        if total_q > remaining_task:
            scale = remaining_task / total_q
            # quants = (quants * scale).int()
            # Ensure at least 1 if scale > 0 but int() floors it? 
            # Simple approach: Loop and set max
            quants = (quants * scale).int()

        # Track execution for this step
        step_revenue = 0
        step_quant_exec = 0
            
        # 3. Submit Orders
        current_time = self.step_counter # using step as time
        
        for i, q in enumerate(quants):
            if q > 0:
                price = price_levels[i]
                order_id = self.trader_unique_id + self.step_counter * 10 + i
                
                msg = {
                    'price': price,
                    'quantity': q.item(),
                    'orderid': order_id,
                    'traderid': self.trader_unique_id,
                    'time': current_time,
                    'time_ns': 0
                }
                
                if self.task == 'sell':
                    self.bid_orders, self.ask_orders, self.trades, executed_q = self._limit_order(
                        side='sell', 
                        orderside_passive=self.bid_orders, 
                        orderside_active=self.ask_orders, 
                        msg=msg
                    )
                    # If we sold, we gained revenue (Price * Q)
                    # _limit_order implementation assumes match at 'best price'.
                    # We need to capture the exact trade prices from self.trades?
                    # For simplicity in demo, assume matched at limit price if aggressive, or monitor trade array diffs.
                    # Simplified: Revenue ~ price * executed_q
                    step_revenue += price * executed_q
                    step_quant_exec += executed_q
                    
                else: # Buy
                     self.ask_orders, self.bid_orders, self.trades, executed_q = self._limit_order(
                        side='buy', 
                        orderside_passive=self.ask_orders, 
                        orderside_active=self.bid_orders, 
                        msg=msg
                    )
                     step_revenue -= price * executed_q # Cost
                     step_quant_exec += executed_q

        # 4. Update State/Rewards
        self.quant_executed += step_quant_exec
        self.total_revenue += step_revenue
        self.step_counter += 1
        
        done = (self.step_counter >= self.max_steps) or (self.quant_executed >= self.task_size)
        
        # Calculate Slippage (Cost)
        # Sell: Arrival Mid - Avg Exec Price
        # Buy: Avg Exec Price - Arrival Mid
        # Normalized by Arrival Mid often, or raw. JAX-LOB usually uses raw or bps.
        
        cost = 0.0
        if step_quant_exec > 0:
            avg_exec_price = step_revenue / step_quant_exec if self.task == 'sell' else -step_revenue / step_quant_exec
            if self.task == 'sell':
                slippage = self.arrival_mid_price - avg_exec_price
            else:
                slippage = avg_exec_price - self.arrival_mid_price
            
            cost = slippage
            
        # Reward
        # Pure Revenue maximization (or Cost Minimization)
        # Reward = Revenue Improvement vs Arrival?
        # Standard: VWAP or Arrival Price.
        # Let's use: Reward = -Slippage (maximize negative slippage)
        # Or simple: Reward = Revenue (for Sell)
        
        if self.task == 'sell':
            reward = float(step_revenue)
        else:
            reward = float(step_revenue) # (negative cost)
            
        # Add a penalty for not finishing?
        if done and self.quant_executed < self.task_size:
            penalty = (self.task_size - self.quant_executed) * self.init_price * 0.1
            reward -= float(penalty) # Big penalty, scaled
        
        info = {
            'quant_executed': self.quant_executed,
            'cost': cost # For Constrained RL
        }
        
        obs = self._get_obs()
        return obs, reward, done, info


    def _limit_order(self, side, orderside_passive, orderside_active, msg):
        """
        Executes a limit order:
        1. Tries to match against passive side (counterparty).
        2. Adds remainder to active side (own side).
        """
        q_executed = 0
        
        # match_order logic operates on a single passive order.
        # We need to loop while we can match.
        
        # Check matching condition
        while msg['quantity'] > 0:
            if side == 'sell':
                # Sell matches against Bids. Top Bid must be >= Sell Price.
                top_idx = get_best_bid_idx(orderside_passive) # Helper 
                if top_idx == -1: break
                best_price = orderside_passive[top_idx, 0]
                if best_price < msg['price']: break # No overlap
            else:
                # Buy matches against Asks. Top Ask must be <= Buy Price.
                top_idx = get_best_ask_idx(orderside_passive)
                if top_idx == -1: break
                best_price = orderside_passive[top_idx, 0]
                if best_price > msg['price']: break
            
            # Match
            # data_tuple: (top_order_idx, orderside, qtm, price, trade, agrOID, time, time_ns)
            orderside_passive, qtm_rem, _, self.trades, _, _, _ = match_order(
                (top_idx, orderside_passive, msg['quantity'], msg['price'], self.trades, msg['orderid'], msg['time'], msg['time_ns'])
            )
            
            q_executed += (msg['quantity'] - qtm_rem)
            msg['quantity'] = qtm_rem
            
            if qtm_rem <= 0: break
            
        # Add remainder
        if msg['quantity'] > 0:
            orderside_active = add_order(orderside_active, msg)
            
        return orderside_passive, orderside_active, self.trades, q_executed

    def _get_obs(self):
        # Generate L2 state
        l2_state = get_L2_state(self.ask_orders, self.bid_orders, n_levels=10)
        
        # Flatten and pad to 610
        # l2 state is 40. We need 610. Padded for now.
        obs = torch.zeros(610, device=self.device)
        obs[:40] = l2_state.float()
        
        # Add high level features (mimic JAX env)
        obs[400:404] = torch.tensor([self.step_counter, 0, 0, 0], device=self.device) # Dummy
        
        return obs.cpu().numpy() # Return numpy for standard Gym compliance
