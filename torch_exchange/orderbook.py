import torch
import sys

# Constants
INITID = -9000
MAX_INT = 2_147_483_647

def init_orderside(nOrders=100, device='cpu'):
    return (torch.ones((nOrders, 6), dtype=torch.int32, device=device) * -1)

def add_order(orderside, msg):
    """
    Adds an order to the orderside.
    msg: dict/object with keys: price, quantity, orderid, traderid, time, time_ns
    """
    # Find first empty slot (where price == -1)
    # orderside shape: (N, 6) -> price, quantity, orderid, traderid, time, time_ns
    
    # Check for empty slots
    empty_indices = torch.where(orderside[:, 0] == -1)[0]
    
    if len(empty_indices) == 0:
        # No space left, currently ignoring or could resize (fixed size in JAX)
        return orderside
        
    idx = empty_indices[0]
    
    new_order = torch.tensor([
        msg['price'],
        max(0, msg['quantity']),
        msg['orderid'],
        msg['traderid'],
        msg['time'],
        msg['time_ns']
    ], dtype=torch.int32, device=orderside.device)
    
    orderside[idx] = new_order
    return _removeZeroNegQuant(orderside)

def _removeZeroNegQuant(orderside):
    # Set orders with quantity <= 0 to -1 (empty)
    mask = orderside[:, 1] <= 0
    orderside[mask] = -1
    return orderside

def cancel_order(orderside, msg):
    """
    Cancels quantity from an order.
    """
    # Find order by orderid
    # msg keys: orderid, quantity, price (optional for lookup)
    
    # Priority 1: Match by orderid
    matches = torch.where(orderside[:, 2] == msg['orderid'])[0]
    
    if len(matches) == 0:
        # Priority 2: Match by price and INITID checks (legacy logic from JAX)
        # init_id_match = ((orderside[:, 0] == msg['price']) & (orderside[:, 2] <= INITID))
        matches = torch.where((orderside[:, 0] == msg['price']) & (orderside[:, 2] <= INITID))[0]
    
    if len(matches) > 0:
        idx = matches[0]
        orderside[idx, 1] -= msg['quantity']
        
    return _removeZeroNegQuant(orderside)

def match_order(data_tuple):
    """
    Core matching logic.
    data_tuple: (top_order_idx, orderside, qtm, price, trade, agrOID, time, time_ns)
    """
    top_order_idx, orderside, qtm, price, trade, agrOID, time, time_ns = data_tuple
    
    # Current order quantity
    current_quant = orderside[top_order_idx, 1]
    
    # Quantity to match is min(available, requested)
    match_quant = min(current_quant, qtm)
    
    # Update order quantity
    new_quant = current_quant - match_quant
    orderside[top_order_idx, 1] = new_quant
    
    # Remaining quantity to match for the aggressor
    qtm_remaining = qtm - match_quant
    
    # Record trade
    # Find empty slot in trade array
    empty_indices = torch.where(trade[:, 0] == -1)[0]
    if len(empty_indices) > 0:
        trade_idx = empty_indices[0]
        # trade format: price, quantity, passiveOID, agrOID, time, time_ns
        trade_entry = torch.tensor([
            orderside[top_order_idx, 0], # Price
            match_quant,                 # Quantity Executed
            orderside[top_order_idx, 2], # Passive Order ID
            agrOID,                      # Aggressor Order ID
            time,
            time_ns
        ], dtype=torch.int32, device=trade.device)
        trade[trade_idx] = trade_entry
        
    orderside = _removeZeroNegQuant(orderside)
    
    return (orderside, qtm_remaining, price, trade, agrOID, time, time_ns)

def get_best_bid_idx(orderside):
    # Highest price. If multiple, lowest time (FIFO).
    # prices = orderside[:, 0]
    # valid_mask = prices != -1
    
    # Filter valid orders
    valid_indices = torch.where(orderside[:, 0] != -1)[0]
    if len(valid_indices) == 0:
        return -1
        
    valid_orders = orderside[valid_indices]
    max_price = torch.max(valid_orders[:, 0])
    
    # Candidates with max price
    best_price_indices = valid_indices[torch.where(valid_orders[:, 0] == max_price)[0]]
    
    # Among those, find min time_s
    best_pricing_orders = orderside[best_price_indices]
    min_time_s = torch.min(best_pricing_orders[:, 4])
    
    time_s_indices = best_price_indices[torch.where(best_pricing_orders[:, 4] == min_time_s)[0]]
    
    # Among those, find min time_ns
    time_s_orders = orderside[time_s_indices]
    min_time_ns = torch.min(time_s_orders[:, 5])
    
    final_indices = time_s_indices[torch.where(time_s_orders[:, 5] == min_time_ns)[0]]
    
    return final_indices[0].item()

def get_best_ask_idx(orderside):
    # Lowest price. If multiple, lowest time (FIFO).
    valid_indices = torch.where(orderside[:, 0] != -1)[0]
    if len(valid_indices) == 0:
        return -1
        
    valid_orders = orderside[valid_indices]
    min_price = torch.min(valid_orders[:, 0])
    
    # Candidates with min price
    best_price_indices = valid_indices[torch.where(valid_orders[:, 0] == min_price)[0]]
    
    # Among those, find min time_s
    best_pricing_orders = orderside[best_price_indices]
    min_time_s = torch.min(best_pricing_orders[:, 4])
    
    time_s_indices = best_price_indices[torch.where(best_pricing_orders[:, 4] == min_time_s)[0]]
    
    # Among those, find min time_ns
    time_s_orders = orderside[time_s_indices]
    min_time_ns = torch.min(time_s_orders[:, 5])
    
    final_indices = time_s_indices[torch.where(time_s_orders[:, 5] == min_time_ns)[0]]
    
    return final_indices[0].item()

def get_best_bid_and_ask_inclQuants(asks, bids):
    best_ask_idx = get_best_ask_idx(asks)
    best_bid_idx = get_best_bid_idx(bids)
    
    if best_ask_idx != -1:
        best_ask_price = asks[best_ask_idx, 0]
        # Sum quantity at this price
        best_ask_q = torch.sum(asks[torch.where(asks[:, 0] == best_ask_price)[0], 1])
    else:
        best_ask_price = MAX_INT
        best_ask_q = 0
        
    if best_bid_idx != -1:
        best_bid_price = bids[best_bid_idx, 0]
        best_bid_q = torch.sum(bids[torch.where(bids[:, 0] == best_bid_price)[0], 1])
    else:
        best_bid_price = -1 # or 0? JAX sets -1/min but usually 0 for price? JAX code uses min/max logic.
        # JAX get_best_bid returns max(bids), if empty -1.
        best_bid_q = 0
        
    # JAX returns ints
    return (torch.tensor([best_ask_price, best_ask_q], dtype=torch.int32, device=asks.device),
            torch.tensor([best_bid_price, best_bid_q], dtype=torch.int32, device=bids.device))

def get_L2_state(asks, bids, n_levels=10):
    device = asks.device
    
    # Process Bids
    valid_bids_mask = bids[:, 0] != -1
    if valid_bids_mask.any():
        valid_bids = bids[valid_bids_mask]
        unique_bid_prices = torch.unique(valid_bids[:, 0])
        # Sort descending
        unique_bid_prices = torch.sort(unique_bid_prices, descending=True).values
        top_bid_prices = unique_bid_prices[:n_levels]
        
        bid_quants = []
        for p in top_bid_prices:
            q = torch.sum(valid_bids[valid_bids[:, 0] == p, 1])
            bid_quants.append(q)
        
        # Pad if fewer than n_levels
        padding = n_levels - len(top_bid_prices)
        if padding > 0:
            top_bid_prices = torch.cat((top_bid_prices, torch.ones(padding, device=device, dtype=torch.int32) * -1))
            bid_quants.extend([torch.tensor(0, device=device, dtype=torch.int32)] * padding)
        
        bid_quants = torch.stack(bid_quants) if isinstance(bid_quants[0], torch.Tensor) else torch.tensor(bid_quants, device=device, dtype=torch.int32)
            
    else:
        top_bid_prices = torch.ones(n_levels, device=device, dtype=torch.int32) * -1
        bid_quants = torch.zeros(n_levels, device=device, dtype=torch.int32)

    # Process Asks
    valid_asks_mask = asks[:, 0] != -1
    if valid_asks_mask.any():
        valid_asks = asks[valid_asks_mask]
        unique_ask_prices = torch.unique(valid_asks[:, 0])
        # Sort ascending
        unique_ask_prices = torch.sort(unique_ask_prices, descending=False).values
        top_ask_prices = unique_ask_prices[:n_levels]
        
        ask_quants = []
        for p in top_ask_prices:
            q = torch.sum(valid_asks[valid_asks[:, 0] == p, 1])
            ask_quants.append(q)
            
        padding = n_levels - len(top_ask_prices)
        if padding > 0:
            # JAX uses MAX_INT or -1 padding logic. Usually L2 data has -1 for empty levels.
            top_ask_prices = torch.cat((top_ask_prices, torch.ones(padding, device=device, dtype=torch.int32) * -1))
            ask_quants.extend([torch.tensor(0, device=device, dtype=torch.int32)] * padding)
            
        ask_quants = torch.stack(ask_quants) if isinstance(ask_quants[0], torch.Tensor) else torch.tensor(ask_quants, device=device, dtype=torch.int32)

    else:
        top_ask_prices = torch.ones(n_levels, device=device, dtype=torch.int32) * -1
        ask_quants = torch.zeros(n_levels, device=device, dtype=torch.int32)

    # Combine: [AskPrices, AskQuants, BidPrices, BidQuants] columns
    # JAX get_L2_state stack(..., axis=1) -> (N, 4) then flatten -> (N*4)
    # columns: asks[:,0], asks[:,1], bids[:,0], bids[:,1]
    
    # Make sure we reshape correctly to match (prices, quants)
    asks_stacked = torch.stack((top_ask_prices, ask_quants), dim=1) # (10, 2)
    bids_stacked = torch.stack((top_bid_prices, bid_quants), dim=1) # (10, 2)
    
    # hstack -> (10, 4) -> flatten -> 40
    l2_state = torch.hstack((asks_stacked, bids_stacked)).flatten()
    return l2_state

