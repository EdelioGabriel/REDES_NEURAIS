"""
Este script contém uma pipeline padrão para problemas que envolvam equações de Laplacesubmetidas a condições de contorno retangulares de duas dimensões
As funções forma construídas de maneira genaralizável, ou seja, para permitir que se defina os parâmetros no momento da chamada
dando mais flexibilidade durante a implementação e possibilitando ajustes rápidos.
"""

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

# Definição do local onde o código serpa executado. Por padrão, gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando: {device}')

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
def sample_boundary(N_b, lb, ub, bc_fns, device):
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

# Cálculo do resíduo da PDE
def pde_residual(model, X):
    
    u = model(X)

    grads = torch.autograd.grad(
        outputs=u,
        inputs=X,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = grads[:, 0].unsqueeze(1)
    u_y = grads[:, 1].unsqueeze(1)

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=X,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0][:, 0].unsqueeze(1)

    u_yy = torch.autograd.grad(
        outputs=u_y,
        inputs=X,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True
    )[0][:, 0].unsqueeze(1)

    res = u_xx + u_yy

    return res

def loss_function(model, X_colloc, X_bc, U_bc, w_data, w_pde):
    """
    Calcula a função de perda total da PINN.

    Args:
        model:  rede neural
        X_col:  pontos de colocação, shape (N_c, 2)
        X_bc:   pontos de contorno, shape (4*N_b, 2)
        U_bc:   valores prescritos nas CCs, shape (4*N_b, 1)
        w_data: peso da perda de dados
        w_pde:  peso da perda física

    Retorna:
        loss:      perda total
        loss_data: perda de dados (para monitoramento)
        loss_pde:  perda física (para monitoramento)
    """

    # loss física
    residual = pde_residual(model, X_colloc)
    loss_pde = torch.mean(residual ** 2)

    # loss dados
    U_pred = model(X_bc)
    loss_data = torch.mean((U_pred - U_bc) ** 2)

    # loss total
    loss = w_data * loss_data + w_pde + loss_pde

    return loss, loss_data, loss_pde

def train(model, optimizer, X_col, X_bc, U_bc, n_epochs, w_data=1.0, w_pde=1.0):
    """
    Loop de treinamento da PINN.

    Args:
        model:    rede neural
        optimizer: otimizador
        X_col:    pontos de colocação, shape (N_c, 2)
        X_bc:     pontos de contorno, shape (4*N_b, 2)
        U_bc:     valores prescritos nas CCs, shape (4*N_b, 1)
        n_epochs: número de épocas
        w_data:   peso da perda de dados
        w_pde:    peso da perda física

    Retorna:
        history: dicionário com o histórico de perdas
                 {'loss': [], 'loss_data': [], 'loss_pde': []}
    """

    history = {'loss': [], 'loss_data': [], 'loss_pde': []}

    for epoch in range(n_epochs):

        optimizer.zero_grad()

        loss, loss_data, loss_pde = loss_function(
            model, X_col, X_bc, U_bc, w_data, w_pde
        )

        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['loss_data'].append(loss_data.item())
        history['loss_pde'].append(loss_pde.item())

        if epoch % 100 == 0:
            print(f'Epoch {epoch:05d} | Loss: {loss.item():.2e} | '
                  f'Loss data: {loss_data.item():.2e} | '
                  f'Loss PDE: {loss_pde.item():.2e}')

    return history

def analytical_solution(X):
    """
    Solução analítica da equação de Laplace 2D com as condições de contorno:
        u(x, 0) = 0
        u(x, 1) = sin(pi * x)
        u(0, y) = 0
        u(1, y) = 0

    Args:
        X: tensor de shape (N, 2) com as coordenadas (x, y)

    Retorna:
        u: tensor de shape (N, 1) com os valores analíticos
    """
    x = X[:, 0].unsqueeze(1)
    y = X[:, 1].unsqueeze(1)

    u = (torch.sinh(torch.pi * y) / torch.sinh(torch.tensor(torch.pi))) * torch.sin(torch.pi * x)

    return u