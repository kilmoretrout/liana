# -*- coding: utf-8 -*-
import torch

def tangent_to_lorentz(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Maps Euclidean tangent vectors (B, N, D-1) to the Lorentz hyperboloid (B, N, D).
    """
    # r is the Euclidean norm of the tangent vector
    r = v.norm(p=2, dim=-1, keepdim=True)  # (B, N, 1)
    
    # Time component is cosh(r)
    x_time = torch.cosh(r)  # (B, N, 1)
    
    # Prevent division by zero for the origin vector
    r_safe = torch.clamp(r, min=eps)
    
    # Space component is sinh(r) * (v / r)
    x_space = torch.sinh(r) * (v / r_safe)  # (B, N, D-1)
    
    # If the vector was perfectly at the origin, force the space component to 0
    x_space = torch.where(r < eps, torch.zeros_like(x_space), x_space)
    
    return torch.cat([x_time, x_space], dim=-1) # (B, N, D)

def PdistPoincare(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Computes pairwise Poincaré distances for a batch of points.
    Assumes all points have norm < 1 (Poincaré ball model).
    """
    # x: (B, N, D)
    B, N, D = x.shape

    # 1. Compute pairwise squared Euclidean distances: ||u - v||^2
    # We use the same broadcasting strategy as your L2 example
    x1 = x.unsqueeze(2)  # (B, N, 1, D)
    x2 = x.unsqueeze(1)  # (B, 1, N, D)
    
    # Squared Euclidean distance
    euclidean_sq_dist = (x1 - x2).pow(2).sum(-1)  # (B, N, N)

    # 2. Compute squared norms: ||u||^2 and ||v||^2
    x_sq_norm = x.pow(2).sum(-1)  # (B, N)

    # Numerical Stability: Ensure norms don't exceed 1-eps to prevent div by zero
    x_sq_norm = torch.clamp(x_sq_norm, max=1 - eps)

    # Broadcast norms for pairwise division
    # Term: (1 - ||u||^2)
    norm_term_u = 1 - x_sq_norm.unsqueeze(2)  # (B, N, 1)
    # Term: (1 - ||v||^2)
    norm_term_v = 1 - x_sq_norm.unsqueeze(1)  # (B, 1, N)

    # 3. Compute the Poincaré formula argument
    # arg = 1 + 2 * (||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
    delta = 2 * euclidean_sq_dist / (norm_term_u * norm_term_v)
    
    # 4. Compute arccosh with stability check
    # arccosh(1) = 0, but gradients are infinite (NaN).
    # We clamp the minimum value to 1 + eps.
    x = 1.0 + delta + eps
    dist = torch.log(x + torch.sqrt(x ** 2 - 1.0))

    # 5. Extract upper triangular part
    # Ensure indices are on the correct device
    i, j = torch.triu_indices(N, N, offset=1, device=x.device)
    
    return dist[:, i, j]  # (B, N*(N-1)/2)

def PdistLorentz(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes pairwise Lorentz distances for a batch of points.
    Assumes points are on the upper sheet of the hyperboloid (x[..., 0] >= 1).
    """
    # x: (B, N, D)  where D is the ambient dimension (e.g., 3 for 2D hyperbolic space)
    B, N, D = x.shape
    
    # 1. Split time and space components
    x_time = x[..., 0]   # (B, N)
    x_space = x[..., 1:] # (B, N, D-1)
    
    # 2. Compute pairwise Minkowski inner product: -u0*v0 + u1*v1 + ...
    # Time product: (B, N, 1) @ (B, 1, N) -> (B, N, N)
    time_prod = -torch.bmm(x_time.unsqueeze(2), x_time.unsqueeze(1))
    
    # Space product: (B, N, D-1) @ (B, D-1, N) -> (B, N, N)
    space_prod = torch.bmm(x_space, x_space.transpose(1, 2))
    
    # Total inner product
    inner_prod = time_prod + space_prod  # (B, N, N)
    
    # 3. Compute arccosh with stability check
    # -inner_prod should mathematically be >= 1.0. 
    # Clamp to 1.0 + eps to avoid NaN gradients when distance is 0.
    safe_prod = torch.clamp(-inner_prod, min=1.0 + eps)
    
    # PyTorch's native acosh handles the log(x + sqrt(x^2 - 1)) formulation safely
    dist = torch.acosh(safe_prod)
    
    # 4. Extract upper triangular part
    i, j = torch.triu_indices(N, N, offset=1, device=x.device)
    
    return dist[:, i, j]  # (B, N*(N-1)/2)

def Pdist(x: torch.Tensor) -> torch.Tensor:
    # x: (B, N, D)
    B, N, D = x.shape

    # Compute pairwise absolute differences via broadcasting
    x1 = x.unsqueeze(2)  # (B, N, 1, D)
    x2 = x.unsqueeze(1)  # (B, 1, N, D)

    diff = (x1 - x2).abs().sum(-1)  # (B, N, N) -- L1 distances

    # Extract upper triangular part to mimic torch.pdist
    i, j = torch.triu_indices(N, N, offset=1)
    return diff[:, i, j]  # (B, N*(N-1)/2)

def PdistL2(x: torch.Tensor) -> torch.Tensor:
    # x: (B, N, D)
    B, N, D = x.shape

    # Compute pairwise differences via broadcasting
    x1 = x.unsqueeze(2)  # (B, N, 1, D)
    x2 = x.unsqueeze(1)  # (B, 1, N, D)

    # L2 Distance: sqrt(sum((x1 - x2)^2))
    # We add 1e-12 to prevent NaN gradients when distance is 0
    diff = (x1 - x2).pow(2).sum(-1).add(1e-12).sqrt() # (B, N, N)

    # Extract upper triangular part to mimic torch.pdist
    i, j = torch.triu_indices(N, N, offset=1)
    return diff[:, i, j]  # (B, N*(N-1)/2)