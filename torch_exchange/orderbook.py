import torch
from typing import Tuple, List

# Constants
INITID = -9000
MAX_INT = 2_147_483_647

def init_orderside(nOrders: int = 100, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    return (torch.ones((nOrders, 6), dtype=torch.int32, device=device) * -1)

@torch.jit.script
def _removeZeroNegQuant(orderside: torch.Tensor) -> torch.Tensor:
    # Set orders with quantity <= 0 to -1 (empty)
    # Col 1 is Quantity
    mask = orderside[:, 1] <= 0
    # We must only zero out existing orders, not already empty ones (though setting -1 to -1 is fine)
    # orderside[mask] = -1 # This works in PyTorch
    if mask.any():
        orderside.masked_fill_(mask.unsqueeze(1), -1)
    return orderside

@torch.jit.script
def add_order(orderside: torch.Tensor, 
              price: int, 
              quantity: int, 
              orderid: int, 
              traderid: int, 
              time: int) -> torch.Tensor:
    """
    Adds an order to the orderside.
    Arguments provided as scalars/ints, will be converted to tensor inside if needed or just placed.
    """
    # Find first empty slot (where price == -1)
    # Note: nonzero() or where() is sync on CPU, async on GPU but returns tensor.
    # orderside shape: (N, 6)
    
    empty_indices = torch.where(orderside[:, 0] == -1)[0]
    
    if len(empty_indices) == 0:
        return orderside
        
    idx = empty_indices[0]
    
    # Assign values directly
    orderside[idx, 0] = price
    orderside[idx, 1] = max(0, quantity)
    orderside[idx, 2] = orderid
    orderside[idx, 3] = traderid
    orderside[idx, 4] = time
    orderside[idx, 5] = 0 # time_ns unused
    
    return _removeZeroNegQuant(orderside)

@torch.jit.script
def cancel_order(orderside: torch.Tensor, 
                 orderid: int, 
                 quantity: int, 
                 price: int) -> torch.Tensor:
    """
    Cancels quantity from an order. (Legacy JAX logic)
    """
    INITID = -9000
    # Priority 1: Match by orderid
    matches = torch.where(orderside[:, 2] == orderid)[0]
    
    if len(matches) == 0:
        # Priority 2: Match by price and INITID checks
        matches = torch.where((orderside[:, 0] == price) & (orderside[:, 2] <= INITID))[0]
    
    if len(matches) > 0:
        idx = matches[0]
        # In-place subtraction
        orderside[idx, 1] -= quantity
        
    return _removeZeroNegQuant(orderside)

@torch.jit.script
def match_order(orderside: torch.Tensor, 
                qtm: int, 
                price: int, 
                trade: torch.Tensor, 
                agrOID: int, 
                time: int,
                top_order_idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
    """
    Core matching logic.
    Returns: (orderside, qtm_remaining, price, trade, agrOID, time)
    """
    
    # Current order quantity
    current_quant = orderside[top_order_idx, 1]
    
    # Quantity to match is min(available, requested)
    match_quant = min(int(current_quant.item()), qtm)
    
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
        # We assign element by element to avoid creating a new tensor
        trade[trade_idx, 0] = int(orderside[top_order_idx, 0].item())
        trade[trade_idx, 1] = match_quant
        trade[trade_idx, 2] = int(orderside[top_order_idx, 2].item())
        trade[trade_idx, 3] = agrOID
        trade[trade_idx, 4] = time
        trade[trade_idx, 5] = 0
        
    orderside = _removeZeroNegQuant(orderside)

    # The return type of match_order is Tuple[torch.Tensor, int, int, torch.Tensor, int, int]
    # The original return statement was commented out.
    # Assuming the function should return the updated orderside and other values.
    return orderside, qtm_remaining, price, trade, agrOID, time

@torch.jit.script
def get_best_bid_idx(orderside: torch.Tensor) -> int:
    price_col = orderside[:, 0]
    # Filter valid
    # valid_indices = torch.where(price_col != -1)[0]
    # JIT optimized max
    
    # We want max price.
    # Empty slots are -1. 
    # If all -1, max is -1
    
    val, idx = torch.max(price_col, 0)
    if val.item() == -1:
        return -1
    return int(idx.item())

@torch.jit.script
def get_best_ask_idx(orderside: torch.Tensor) -> int:
    MAX_INT = 2_147_483_647
    price_col = orderside[:, 0]
    
    # We want min price.
    # Empty slots are -1. We should ignore them.
    # Replace -1 with MAX_INT
    
    # temp_prices = torch.where(price_col != -1, price_col, torch.tensor(MAX_INT, dtype=torch.int32, device=orderside.device))
    # Optimize: 
    valid_mask = price_col != -1
    if not valid_mask.any():
        return -1
        
    temp_prices = torch.where(valid_mask, price_col, torch.tensor(MAX_INT, dtype=torch.int32, device=orderside.device))
    
    val, idx = torch.min(temp_prices, 0)
    
    if val.item() == MAX_INT:
        return -1
        
    return int(idx.item())

@torch.jit.script
def get_best_bid_and_ask_inclQuants(ask_orders: torch.Tensor, bid_orders: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    MAX_INT = 2_147_483_647
    # ASKS
    ask_prices = ask_orders[:, 0]
    valid_asks_mask = ask_prices != -1
    
    if valid_asks_mask.any():
        # Mask out invalid (-1) by setting to MAX_INT
        temp_asks = torch.where(valid_asks_mask, ask_prices, torch.tensor(MAX_INT, dtype=torch.int32))
        best_ask_idx = torch.argmin(temp_asks)
        
        best_ask_price = temp_asks[best_ask_idx]
        
        # Sum quantity at this price
        # JIT efficient masking
        # ask_orders[:, 0] == best_ask_price
        # ask_orders[:, 1]
        
        mask_price = (ask_orders[:, 0] == best_ask_price) & valid_asks_mask
        best_ask_q = torch.sum(
            torch.where(mask_price, ask_orders[:, 1], torch.tensor(0, dtype=torch.int32))
        )
             
        # best_ask_price is 0-d tensor, best_ask_q is 0-d tensor
        # Use stack instead of tensor([...]) to avoid "got Tensor" error
        best_ask = torch.stack([best_ask_price, best_ask_q]).to(dtype=torch.int32)
    else:
        best_ask = torch.tensor([MAX_INT, 0], dtype=torch.int32)

    # BIDS
    bid_prices = bid_orders[:, 0]
    valid_bids_mask = bid_prices != -1
    
    if valid_bids_mask.any():
        # -1 is invalid, but also small. Max works fine.
        best_bid_idx = torch.argmax(bid_prices)
        best_bid_price = bid_prices[best_bid_idx]
        
        if best_bid_price == -1: # Should be covered by valid_bids_mask check
             best_bid = torch.tensor([-1, 0], dtype=torch.int32, device=bid_orders.device)
        else:
             mask_price = (bid_orders[:, 0] == best_bid_price) & valid_bids_mask
             best_bid_q = torch.sum(torch.where(mask_price, bid_orders[:, 1], torch.tensor(0, dtype=torch.int32, device=bid_orders.device)))
             best_bid = torch.stack([best_bid_price, best_bid_q]).to(dtype=torch.int32)
    else:
        best_bid = torch.tensor([-1, 0], dtype=torch.int32, device=bid_orders.device)
        
    return best_ask, best_bid

@torch.jit.script
def get_L2_state(ask_orders: torch.Tensor, bid_orders: torch.Tensor, n_levels: int = 10) -> torch.Tensor:
    MAX_INT = 2_147_483_647
    device = ask_orders.device
    
    # Process Bids
    valid_bids_mask = bid_orders[:, 0] != -1
    if valid_bids_mask.any():
        # JIT workaround: slicing boolean mask might be tricky.
        # use masked_select?
        valid_bids = bid_orders[valid_bids_mask]
        
        # unique
        unique_bid_prices = torch.unique(valid_bids[:, 0])
        # Sort descending
        unique_bid_prices = torch.sort(unique_bid_prices, descending=True).values
        top_bid_prices = unique_bid_prices[:n_levels]
        
        # We need to construct output tensor.
        # Loop is acceptable for N=10 in JIT
        
        # bid_quants = torch.zeros(n_levels, dtype=torch.int32, device=device)
        # Using list then stacking is standard
        bid_quants_list: List[torch.Tensor] = []
        
        for i in range(len(top_bid_prices)):
            p = top_bid_prices[i]
            # Sum quantity
            # valid_bids is (M, 6)
            mask = valid_bids[:, 0] == p
            # q = torch.sum(valid_bids[mask, 1]) # This slicing might be slow?
            # masked_select is better?
            # q = torch.sum(valid_bids[:, 1].masked_select(mask))
            # simple where
            q = torch.sum(torch.where(mask, valid_bids[:, 1], torch.tensor(0, device=device, dtype=torch.int32)))
            bid_quants_list.append(q)
            
        # Pad
        padding = n_levels - len(top_bid_prices)
        if padding > 0:
            # We need to return Prices AND Quants? 
            # The original just returned quants?
            # Wait, get_L2_state typically returns [Price, Quant, Price, Quant...] flattened?
            # L2 usually [P1, Q1, P2, Q2 ...]
            
            # Let's check environment usage: obs[:40] = l2_state.float()
            # So 40 items. 10 levels * 2 (P, Q) * 2 (Bid, Ask) = 40. Yes.
            
            # Pad Prices
            pad_p = torch.ones(padding, device=device, dtype=torch.int32) * -1
            top_bid_prices = torch.cat((top_bid_prices, pad_p))
            
            # Pad Quants
            for _ in range(padding):
                bid_quants_list.append(torch.tensor(0, device=device, dtype=torch.int32))
                
        # Stack quants
        bid_quants = torch.stack(bid_quants_list)
        
    else:
        top_bid_prices = torch.ones(n_levels, device=device, dtype=torch.int32) * -1
        bid_quants = torch.zeros(n_levels, device=device, dtype=torch.int32)

    # Process Asks
    valid_asks_mask = ask_orders[:, 0] != -1
    if valid_asks_mask.any():
        valid_asks = ask_orders[valid_asks_mask]
        unique_ask_prices = torch.unique(valid_asks[:, 0])
        unique_ask_prices = torch.sort(unique_ask_prices, descending=False).values
        top_ask_prices = unique_ask_prices[:n_levels]
        
        ask_quants_list: List[torch.Tensor] = []
        
        for i in range(len(top_ask_prices)):
            p = top_ask_prices[i]
            mask = valid_asks[:, 0] == p
            q = torch.sum(torch.where(mask, valid_asks[:, 1], torch.tensor(0, device=device, dtype=torch.int32)))
            ask_quants_list.append(q)
            
        padding = n_levels - len(top_ask_prices)
        if padding > 0:
            pad_p = torch.tensor([MAX_INT]*padding, device=device, dtype=torch.int32)
            top_ask_prices = torch.cat((top_ask_prices, pad_p))
            for _ in range(padding):
                ask_quants_list.append(torch.tensor(0, device=device, dtype=torch.int32))
                
        ask_quants = torch.stack(ask_quants_list)
    else:
        top_ask_prices = torch.ones(n_levels, device=device, dtype=torch.int32) * MAX_INT
        ask_quants = torch.zeros(n_levels, device=device, dtype=torch.int32)
        
    # Interleave [AskP, AskQ, BidP, BidQ] or [AskP ... BidP ...]? 
    # Usually: Asks [P, Q, P, Q], Bids [P, Q, P, Q]
    # Let's stack [Asks, Bids]
    
    # Asks: 20 elements (10 P, 10 Q)
    asks_flat = torch.stack((top_ask_prices, ask_quants), dim=1).flatten()
    bids_flat = torch.stack((top_bid_prices, bid_quants), dim=1).flatten()
    
    return torch.cat((asks_flat, bids_flat))
