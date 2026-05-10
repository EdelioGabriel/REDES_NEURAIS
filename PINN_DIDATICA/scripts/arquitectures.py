"""
Este script contém as funções necessárias par aa discussão do notebook "08_training_validation.ipynb"

Elas são particulares para a aplicação de de uma ResNet híbrida com PINN
"""

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import time

class ResidualBlock(nn.Module):
    def __init__(self, n_hidden, activation):
        super().__init__()
        self.linear1 = nn.Linear(n_hidden, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.activation = activation()

    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)  # skip connection


class ResNetPINN(nn.Module):
    """
    Arquitetura residual para PINNs.
    Mesma assinatura que PINN para facilitar comparação direta.

    n_layers aqui define o número de blocos residuais.
    Cada bloco contém 2 lineares + skip connection.
    """
    def __init__(self, n_inputs, n_outputs, n_hidden, n_layers, activation):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            activation()
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_hidden, activation) for _ in range(n_layers)]
        )

        self.output_layer = nn.Linear(n_hidden, n_outputs)

    def forward(self, X):
        X = self.input_layer(X)
        X = self.res_blocks(X)
        return self.output_layer(X)

def u_exact(x, y):
    return (0.8 * torch.sin(torch.pi/2 * (x + 0.78)) * torch.cos(y - 1)
          - 0.8 * torch.sin(torch.pi/2 * (x + 1.50)) * torch.cos(y + 1))

def v_exact(x, y):
    return 0.72 - 0.65 * torch.exp(-x**2 * y**2 / 2 + x)

def sample_boundary_rectangular_elasticity(N_b, lb, ub, bc_fns, device):
    """
    Amostra N_b pontos em cada face do domínio [lb, ub]²
    para problemas de elasticidade 2D (duas saídas: u e v).

    Args:
        N_b:    número de pontos por face
        lb:     limite inferior do domínio — lista [lb_x, lb_y]
        ub:     limite superior do domínio — lista [ub_x, ub_y]
        bc_fns: dicionário com funções de CC para cada face:
                {'bottom': fn, 'top': fn, 'left': fn, 'right': fn}
                cada fn recebe um tensor 1D e retorna tensor (N, 2) com [u, v]
        device: dispositivo de execução

    Retorna:
        X_bc: tensor (4*N_b, 2) — coordenadas
        U_bc: tensor (4*N_b, 2) — valores [u, v] nas bordas
    """
    lb = torch.tensor(lb, dtype=torch.float32, device=device)
    ub = torch.tensor(ub, dtype=torch.float32, device=device)

    lb_x, lb_y = lb[0], lb[1]
    ub_x, ub_y = ub[0], ub[1]

    t = torch.rand(N_b, device=device) * (ub_x - lb_x) + lb_x
    s = torch.rand(N_b, device=device) * (ub_y - lb_y) + lb_y

    # face inferior: y = lb_y, x livre
    X_bottom = torch.stack([t, torch.full_like(t, lb_y)], dim=1)
    U_bottom = bc_fns['bottom'](t)   # shape (N_b, 2)

    # face superior: y = ub_y, x livre
    X_top = torch.stack([t, torch.full_like(t, ub_y)], dim=1)
    U_top = bc_fns['top'](t)

    # face esquerda: x = lb_x, y livre
    X_left = torch.stack([torch.full_like(s, lb_x), s], dim=1)
    U_left = bc_fns['left'](s)

    # face direita: x = ub_x, y livre
    X_right = torch.stack([torch.full_like(s, ub_x), s], dim=1)
    U_right = bc_fns['right'](s)

    X_bc = torch.cat([X_bottom, X_top, X_left, X_right], dim=0)
    U_bc = torch.cat([U_bottom, U_top, U_left, U_right], dim=0)

    return X_bc, U_bc

def compute_forcing_elasticity(X_col, lam, mu):
    """
    Calcula os termos de força fx e fy a partir da solução manufaturada
    (Equação 21 do artigo) via autodiferenciação.

    A ideia: substituímos u_exact e v_exact nas equações de Navier e
    isolamos fx e fy — o que sobra do lado direito.

    Args:
        X_col: pontos de colocação — shape (N, 2), requires_grad=True
        lam:   constante de Lamé λ
        mu:    constante de Lamé μ

    Retorna:
        fx: tensor (N, 1)
        fy: tensor (N, 1)
    """
    x, y = X_col[:, 0:1], X_col[:, 1:2]

    u = u_exact(x, y)
    v = v_exact(x, y)

    # ── Derivadas de primeira ordem ──────────────────────────────────────────
    u_x = torch.autograd.grad(u.sum(), X_col, create_graph=True)[0][:, 0:1]
    u_y = torch.autograd.grad(u.sum(), X_col, create_graph=True)[0][:, 1:2]
    v_x = torch.autograd.grad(v.sum(), X_col, create_graph=True)[0][:, 0:1]
    v_y = torch.autograd.grad(v.sum(), X_col, create_graph=True)[0][:, 1:2]

    # ── Derivadas de segunda ordem ───────────────────────────────────────────
    u_xx = torch.autograd.grad(u_x.sum(), X_col, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), X_col, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x.sum(), X_col, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y.sum(), X_col, create_graph=True)[0][:, 1:2]

    # ── Divergência e seus gradientes ────────────────────────────────────────
    div   = u_x + v_y
    div_x = torch.autograd.grad(div.sum(), X_col, create_graph=True)[0][:, 0:1]
    div_y = torch.autograd.grad(div.sum(), X_col, create_graph=True)[0][:, 1:2]

    # ── Isolando fx e fy das equações de Navier ──────────────────────────────
    # N1 = (λ+μ)*div_x + μ*(u_xx + u_yy) + fx = 0  =>  fx = -N1
    # N2 = (λ+μ)*div_y + μ*(v_xx + v_yy) + fy = 0  =>  fy = -N2
    fx = -((lam + mu) * div_x + mu * (u_xx + u_yy))
    fy = -((lam + mu) * div_y + mu * (v_xx + v_yy))

    return fx.detach(), fy.detach()

def elasticity_loss(model, X_col, fx, fy, X_bc, U_bc, lam, mu, lambda_bc=10.0):
    """
    Loss para elasticidade 2D (Equações de Navier).

    J = J_pde + lambda_bc * J_bc

    Args:
        model:      rede neural (PINN ou ResNetPINN)
        X_col:      pontos de colocação internos — shape (N, 2), requires_grad=True
        fx:         força externa em x — shape (N, 1), pré-computada
        fy:         força externa em y — shape (N, 1), pré-computada
        X_bc:       pontos de contorno — shape (M, 2)
        U_bc:       valores de CC — shape (M, 2), colunas [u, v]
        lam:        constante de Lamé λ
        mu:         constante de Lamé μ
        lambda_bc:  peso da loss de contorno

    Retorna:
        loss, J_pde, J_bc  (para logging)
    """
    # ── Predição nos pontos internos ─────────────────────────────────────────
    UV    = model(X_col)
    u, v  = UV[:, 0:1], UV[:, 1:2]

    # ── Derivadas de primeira ordem ──────────────────────────────────────────
    u_x = torch.autograd.grad(u.sum(), X_col, create_graph=True)[0][:, 0:1]
    u_y = torch.autograd.grad(u.sum(), X_col, create_graph=True)[0][:, 1:2]
    v_x = torch.autograd.grad(v.sum(), X_col, create_graph=True)[0][:, 0:1]
    v_y = torch.autograd.grad(v.sum(), X_col, create_graph=True)[0][:, 1:2]

    # ── Derivadas de segunda ordem ───────────────────────────────────────────
    u_xx = torch.autograd.grad(u_x.sum(), X_col, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), X_col, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x.sum(), X_col, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y.sum(), X_col, create_graph=True)[0][:, 1:2]

    # ── Divergência e seus gradientes ────────────────────────────────────────
    div   = u_x + v_y
    div_x = torch.autograd.grad(div.sum(), X_col, create_graph=True)[0][:, 0:1]
    div_y = torch.autograd.grad(div.sum(), X_col, create_graph=True)[0][:, 1:2]

    # ── Resíduos das equações de Navier ──────────────────────────────────────
    R1 = (lam + mu) * div_x + mu * (u_xx + u_yy) + fx
    R2 = (lam + mu) * div_y + mu * (v_xx + v_yy) + fy
    J_pde = (R1**2 + R2**2).mean()

    # ── Loss de contorno ─────────────────────────────────────────────────────
    UV_bc = model(X_bc)
    J_bc  = ((UV_bc[:, 0:1] - U_bc[:, 0:1])**2 +
             (UV_bc[:, 1:2] - U_bc[:, 1:2])**2).mean()

    loss = J_pde + lambda_bc * J_bc
    return loss, J_pde.item(), J_bc.item()

def train(model, loss_fn, optimizer, n_iter, log_every=500):
    """
    Loop de treino genérico para PINNs.

    Args:
        model:      rede neural (PINN ou ResNetPINN)
        loss_fn:    função que recebe o model e retorna (loss, *extras)
                    ex: lambda model: elasticity_loss(model, X_col, ...)
        optimizer:  otimizador já instanciado
        n_iter:     número de iterações
        log_every:  frequência de impressão do log

    Retorna:
        history: lista de dicts com 'iter', 'loss', 'time' e quaisquer
                 extras retornados pela loss_fn
    """
    history = []

    for it in range(1, n_iter + 1):
        optimizer.zero_grad()
        loss, *extras = loss_fn(model)
        loss.backward()
        optimizer.step()

        history.append({
            'iter':  it,
            'loss':  loss.item(),
            'J_pde': extras[0],
            'J_bc':  extras[1],
        })

        if it % log_every == 0:
            print(f'Iter {it:5d} | Loss: {loss.item():.2e} | J_pde: {extras[0]:.2e} | J_bc: {extras[1]:.2e} ')

    return history