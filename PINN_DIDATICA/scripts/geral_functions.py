"""
Este script contém todas as funções e estruturas gerais para a implementação dos exemplos, evitando repetição desnecessária

- Nota do autor
"""

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

# Classe de construção da arquitetura da rede - tipo MLP (Multilayer Perceptron)
class PINN(nn.Module):
    """
    Classe para definir a arquitetura MLP
    Recebe como parâmetros de arquitetura:

        n_inputs: Quantidade de dados de entrada
        n_outputs: Quantidade de dados de saída
        n_hidden: Quantidade de neurônios nas camadas ocultas
        n_layers: Quantidade de camadas
        activation: Função de ativação

    Retorna: O vetor X após o passo forward
    """
    def __init__(self, n_inputs, n_outputs, n_hidden, n_layers, activation):
        super().__init__()

        layers = []

        layers.append(nn.Linear(n_inputs, n_hidden))
        layers.append(activation())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_hidden, n_hidden))
            layers.append(activation())

        layers.append(nn.Linear(n_hidden, n_outputs))

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        X = self.net(X)
        return X
    
# Função de amostragem dos pontos de colocação - internos    
def sample_collocation(N_c, lb, ub, device):
    """
    Amostra N_c pontos aleatórios no interior do domínio [lb, ub]^2.
    
    Args:
        N_c:    número de pontos de colocação
        lb:     limite inferior do domínio (float)
        ub:     limite superior do domínio (float)
        device: dispositivo de execução (cpu ou cuda)

    Retorna tensor de shape (N_c, 2) com requires_grad=True.
    """
    X = torch.rand(N_c, 2, device=device) * (ub - lb) + lb
    X.requires_grad_(True)
    return X

# Função para amostragem dos pontos no domínio
def sample_boundary_rectangular(N_b, lb, ub, bc_fns, device):
    """
    Amostra N_b pontos em cada face do domínio [lb, ub]^2 e calcula
    os valores de CC correspondentes.

    Args:
        N_b:    número de pontos por face
        lb:     limite inferior do domínio (float)
        ub:     limite superior do domínio (float)
        bc_fns: dicionário com as funções de CC para cada face:
                {'bottom': fn, 'top': fn, 'left': fn, 'right': fn}
                cada fn recebe um tensor 1D e retorna um tensor 1D
        device: dispositivo de execução (cpu ou cuda)

    Retorna:
        X_bc: tensor de shape (4 * N_b, 2) com as coordenadas
        U_bc: tensor de shape (4 * N_b, 1) com os valores de CC
    """
    t = torch.rand(N_b, device=device) * (ub - lb) + lb

    # face inferior: y = lb, x livre
    X_bottom = torch.stack([t, torch.full_like(t, lb)], dim=1)
    U_bottom = bc_fns['bottom'](t).unsqueeze(1)

    # face superior: y = ub, x livre
    X_top = torch.stack([t, torch.full_like(t, ub)], dim=1)
    U_top = bc_fns['top'](t).unsqueeze(1)

    # face esquerda: x = lb, y livre
    X_left = torch.stack([torch.full_like(t, lb), t], dim=1)
    U_left = bc_fns['left'](t).unsqueeze(1)

    # face direita: x = ub, y livre
    X_right = torch.stack([torch.full_like(t, ub), t], dim=1)
    U_right = bc_fns['right'](t).unsqueeze(1)

    X_bc = torch.cat([X_bottom, X_top, X_left, X_right], dim=0)
    U_bc = torch.cat([U_bottom, U_top, U_left, U_right], dim=0)

    return X_bc, U_bc