"""
arquitectures.py

Funções para o notebook 06_architectures.ipynb.
Comparação entre MLP vanilla e ResNet-PINN aplicadas
ao problema de Helmholtz 2D — arcada magnética solar.

A motivação para ResNet é a profundidade necessária para capturar
as oscilações espaciais da solução — onde o vanishing gradient
da MLP vanilla dificulta o treinamento.

Referência:
    - Zhang et al. (2021), Water, 13(4), 423.
    - Baty, H. (2024). arXiv:2403.00599
"""

import torch
import torch.nn as nn

# ── Bloco residual ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Bloco residual com skip connection.

    Realiza:
        out = activation(linear2(activation(linear1(x))) + x)

    A skip connection permite que o gradiente flua diretamente
    pelas camadas mais profundas, evitando o problema de
    vanishing gradient em redes profundas.

    Args:
        n_hidden:   número de neurônios
        activation: classe da função de ativação
    """
    def __init__(self, n_hidden, activation):
        super().__init__()
        self.linear1    = nn.Linear(n_hidden, n_hidden)
        self.linear2    = nn.Linear(n_hidden, n_hidden)
        self.activation = activation()

    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return self.activation(x + residual)

# ── ResNet-PINN ────────────────────────────────────────────────────────────────

class ResNetPINN(nn.Module):
    """
    Arquitetura residual para PINNs.
    Mesma assinatura que PINN para facilitar comparação direta.

    n_layers define o número de blocos residuais.
    Cada bloco contém 2 camadas lineares + skip connection,
    então a profundidade efetiva é 2 * n_layers + 1.

    Args:
        n_inputs:   dimensão da entrada
        n_outputs:  dimensão da saída
        n_hidden:   número de neurônios por camada
        n_layers:   número de blocos residuais
        activation: classe da função de ativação
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
#