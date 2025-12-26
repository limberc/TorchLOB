import torch
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from typing import Dict, Tuple, Optional, List
from torch_exchange.orderbook import (
    init_orderside, add_order, cancel_order, match_order, 
    get_best_bid_and_ask_inclQuants, get_L2_state,
    get_best_bid_idx, get_best_ask_idx
)
import pdb  

# Constants
MAX_INT = 2_147_483_647

class TorchExecutionEnv(gym.Env):
    """
    A PyTorch-based Limit Order Book (LOB) Execution Environment.
    (Gymnasium Compatible)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, task='sell', task_size=5000, device='cpu',
    slice_time_window=1800, book_depth=10, tick_size=1, init_price=100000, data_path=None, nrows=10000):
        super(TorchExecutionEnv, self).__init__()
        self.device = device
        self.task = task
        self.task_size = task_size
        self.n_actions = book_depth
        self.tick_size = tick_size
        self.n_ticks_in_book = 2
        
        # Action space
        self.action_space = spaces.Box(low=0, high=10, shape=(self.n_actions,), dtype=int)
        
        # Observation space
        obs_dim = book_depth * 4
        self.observation_space = spaces.Box(low=float('-inf'), high=float('inf'), shape=(obs_dim,), dtype=float)

        # Internal State placeholders
        self.state = None
        
        # Mock Data Parameters
        self.sliceTimeWindow = slice_time_window
        self.stepLines = 100
        self.nOrdersPerSide = 100
        self.nTradesLogged = 100
        self.book_depth = book_depth
        self.trader_unique_id = -9000 + 1
        
        # Initial Price 
        self.init_price = init_price
        self.arrival_mid_price = self.init_price
        
        # Normalization
        self.initial_value = self.task_size * self.init_price
        
        # Load Real Data
        self.data_path = data_path
        self.price_history = None
        if data_path is not None:
            print(f"Loading data from {data_path}...")
            # Assuming 'midpoint' column exists. If strictly no header, use index 2.
            # Based on inspection, it has header 'midpoint' or is col 2.
            # Let's try header inference.
            try:
                df = pd.read_csv(data_path, nrows=nrows)
                if 'midpoint' in df.columns:
                    prices = df['midpoint'].values
                else:
                    # Fallback to column 2
                    prices = df.iloc[:, 2].values
            except Exception as e:
                print(f"Error loading data: {e}. Falling back to random walk.")
                prices = None
                
            if prices is not None:
                self.price_history = torch.tensor(prices, device=device, dtype=torch.float32)
                self.init_price = self.price_history[0].item()
                self.price_len = len(self.price_history)
                print(f"Loaded {self.price_len} price points.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # State tracking
        self.quant_executed = 0
        self.task_to_execute = self.task_size
        self.step_counter = 0
        self.max_steps = 100
        self.total_revenue = 0
        self.time = 0
        
        # Initialize Order Book State
        self.ask_orders = init_orderside(self.nOrdersPerSide, self.device)
        self.bid_orders = init_orderside(self.nOrdersPerSide, self.device)
        self.trades = (torch.ones((self.nTradesLogged, 6), dtype=torch.int32, device=self.device) * -1)
        
        # Reset Price from History
        if self.price_history is not None:
             max_start = self.price_len - int(self.sliceTimeWindow) - 100
             if max_start > 0: self.data_idx = torch.randint(0, max_start, (1,)).item()
             else: self.data_idx = 0
             self.arrival_mid_price = self.price_history[self.data_idx].item()
        else:
            self.arrival_mid_price = self.init_price

        # Populate LOB
        self._regenerate_lob()
        
        # Track Arrival Mid Price
        # Re-calculated by _regenerate_lob above, but consistency check
        # best_ask, best_bid = get_best_bid_and_ask_inclQuants(self.ask_orders, self.bid_orders)
        # ...
        # logic removed/superseded by explicit arrival_mid_price

        return self._get_obs(), {}

    def step(self, action):
        """
        action: array of shape (4,) representing quantities to submit at [FT, M, NT, PP] prices.
        """
        # 0. Update Market State (Data Replay / Drift)
        if self.price_history is not None:
             self.data_idx += 1
             if self.data_idx < self.price_len:
                 self.arrival_mid_price = self.price_history[self.data_idx].item()
        else:
             drift = torch.randn(1).item() * 5
             self.arrival_mid_price += drift
             
        # Regenerate LOB around new Price
        self._regenerate_lob()

        # 1. Determine Prices
        # Get BBO
        best_ask, best_bid = get_best_bid_and_ask_inclQuants(self.ask_orders, self.bid_orders)
        best_ask_price = best_ask[0].item()
        best_bid_price = best_bid[0].item()
        
        if best_bid_price == -1: best_bid_price = self.arrival_mid_price - 100
        if best_ask_price == MAX_INT: best_ask_price = self.arrival_mid_price + 100

        # Define Price Levels logic
        # FT = Far Touch
        # M = Mid
        # NT = Near Touch
        # PP = Passive
        
        # Determine target price levels based on n_actions
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
        # action is quantities (Tensor)
        if isinstance(action, torch.Tensor):
            quants = action.to(dtype=torch.int32, device=self.device)
        else:
            quants = torch.tensor(action, dtype=torch.int32, device=self.device)
             
        # Clip to task size
        remaining_task = self.task_to_execute - self.quant_executed
        total_q = quants.sum()
        
        if total_q > remaining_task:
            scale = remaining_task / total_q
            # Clip quantities to match remaining task
            quants = (quants * scale).int()

        # Track execution for this step
        step_revenue = 0
        step_quant_exec = 0
            
        # 3. Submit Orders
        current_time = self.step_counter # using step as time
        
        for i, q in enumerate(quants):
            if q > 0:
                price = price_levels[i]
                orderid = self.trader_unique_id + self.step_counter * 10 + i
                
                if self.task == 'sell':
                    self.bid_orders, self.ask_orders, self.trades, executed_q = self._limit_order(
                        'sell', 
                        self.bid_orders, 
                        self.ask_orders, 
                        int(price),
                        int(q.item()),
                        orderid,
                        self.trader_unique_id,
                        int(current_time)
                    )
                    # Revenue ~ price * executed_q
                    step_revenue += price * executed_q
                    step_quant_exec += executed_q
                    
                else: # Buy
                     self.ask_orders, self.bid_orders, self.trades, executed_q = self._limit_order(
                        'buy', 
                        self.ask_orders, 
                        self.bid_orders, 
                        int(price),
                        int(q.item()),
                        orderid,
                        self.trader_unique_id,
                        int(current_time)
                    )
                     step_revenue -= price * executed_q # Cost
                     step_quant_exec += executed_q

        # 4. Update State/Rewards
        self.quant_executed += step_quant_exec
        self.total_revenue += step_revenue
        self.step_counter += 1
        
        terminated = (self.quant_executed >= self.task_size)
        truncated = (self.step_counter >= self.max_steps)
        done = terminated or truncated
        
        # Calculate Slippage (Cost)
        cost = 0.0
        if step_quant_exec > 0:
            avg_exec_price = step_revenue / step_quant_exec if self.task == 'sell' else -step_revenue / step_quant_exec
            if self.task == 'sell':
                slippage = self.arrival_mid_price - avg_exec_price
            else:
                slippage = avg_exec_price - self.arrival_mid_price
            
            cost = slippage
            
        # Reward: Implementation Shortfall / Value Realized
        # Normalize by Initial Value to get decent scale [0, 1]
        reward = float(step_revenue) / self.initial_value
        
        # Penalty for not finishing
        if done and self.quant_executed < self.task_size:
            # Simple hefty penalty: 
            rem_value = (self.task_size - self.quant_executed) * self.init_price
            penalty = rem_value / self.initial_value
            reward -= penalty # Effectively subtracting the un-realized value
        
        info = {
            'quant_executed': self.quant_executed,
            'cost': cost # For Constrained RL
        }
        
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info


    def _regenerate_lob(self):
        # Clear Books
        self.ask_orders = init_orderside(self.nOrdersPerSide, self.device)
        self.bid_orders = init_orderside(self.nOrdersPerSide, self.device)
        
        # Re-populate around current arrival_mid_price
        price = self.arrival_mid_price
        
        for i in range(10):
            # Add Ask
            self.ask_orders = add_order(
                self.ask_orders,
                int(price + (i + 1) * 100),
                100, i + 1000 + self.step_counter * 100, 0, 0
            )
            # Add Bid
            self.bid_orders = add_order(
                self.bid_orders,
                int(price - (i + 1) * 100),
                100, i + 2000 + self.step_counter * 100, 0, 0
            )

    def _limit_order(self, side, orderside_passive, orderside_active, price, quantity, orderid, traderid, time):
        """
        Executes a limit order via JIT primitives.
        """
        q_executed = 0
        current_q = quantity
        
        # match_order logic operates on a single passive order.
        # Loop while we have quantity to execute
        while current_q > 0:
            if side == 'sell':
                # Sell matches against Bids (Passive)
                top_idx = get_best_bid_idx(orderside_passive) 
                if top_idx == -1: break
                
                # Check Price Condition: Best Bid >= Sell Price
                # JIT function returns int index, we access tensor
                best_price = int(orderside_passive[top_idx, 0].item())
                if best_price < price: break 
            else:
                # Buy matches against Asks (Passive)
                top_idx = get_best_ask_idx(orderside_passive)
                if top_idx == -1: break
                
                # Check Price Condition: Best Ask <= Buy Price
                best_price = int(orderside_passive[top_idx, 0].item())
                if best_price > price: break
            
            # Match
            # match_order(orderside, qtm, price, trade, agrOID, time, top_order_idx)
            # Returns: (orderside, qtm_remaining, price, trade, agrOID, time)
            
            orderside_passive, qtm_rem, _, self.trades, _, _ = match_order(
                orderside_passive,
                current_q,
                price,
                self.trades,
                orderid,
                time,
                top_idx
            )
            
            matched = current_q - qtm_rem
            q_executed += matched
            current_q = qtm_rem
            
            if qtm_rem <= 0: break
            
        # Add remainder
        if current_q > 0:
            orderside_active = add_order(
                orderside_active,
                price,
                current_q,
                orderid,
                traderid,
                time
            )
            
        return orderside_passive, orderside_active, self.trades, q_executed

    def _get_obs(self):
        # Generate L2 state
        return get_L2_state(self.ask_orders, self.bid_orders, n_levels=self.book_depth).float()
