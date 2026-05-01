"""
plot_utils.py — funções de visualização para notebooks de PINNs.

Todas as funções recebem arrays numpy prontos — nenhuma dependência de torch,
modelos ou funções analíticas. A conversão de tensores para arrays é
responsabilidade do script particular de cada exemplo.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Dimensões e fonte ──────────────────────────────────────────────────────────

FONT_SIZE   = 16
FONT_FAMILY = "Arial"
SQ_SIZE     = 500

AXIS_COMMON = dict(
    showline=True,
    linewidth=1.5,
    linecolor="black",
    mirror=True,
    ticks="inside",
    ticklen=5,
    tickwidth=1.5,
    tickcolor="black",
)

LAYOUT_BASE = dict(
    template="simple_white",
    font=dict(family=FONT_FAMILY, size=FONT_SIZE),
    margin=dict(l=80, r=60, t=70, b=70),
)

COLORSCALE = "RdBu_r"
COLORS     = ["#2C3E50", "#E74C3C", "#2980B9", "#27AE60"]

# ── Funções auxiliares ─────────────────────────────────────────────────────────

def _square_axis(title, log=False):
    """Retorna dict de eixo com borda e fonte ABNT."""
    ax = dict(**AXIS_COMMON, title=dict(text=title, font=dict(size=FONT_SIZE)))
    if log:
        ax["type"] = "log"
    return ax


# ── 1. Curvas de aprendizado ───────────────────────────────────────────────────

def plot_loss(history):
    """
    Plota as curvas de aprendizado disponíveis no history.
    Aceita qualquer combinação de: 'loss', 'loss_data', 'loss_pde'

    Args:
        history: dicionário {'loss': [], 'loss_data': [], 'loss_pde': []}
    """
    labels = {
        'loss':      ('Total', '#2C3E50', 'solid'),
        'loss_data': ('Dados', '#E74C3C', 'dash'),
        'loss_pde':  ('PDE',   '#2980B9', 'dot'),
    }

    epochs = list(range(len(next(iter(history.values())))))

    fig = go.Figure()

    for key, (name, color, dash) in labels.items():
        if key in history:
            fig.add_trace(go.Scatter(
                x=epochs, y=history[key],
                mode='lines', name=name,
                line=dict(color=color, width=2, dash=dash)
            ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Curvas de aprendizado", font=dict(size=FONT_SIZE + 2)),
        xaxis=_square_axis("Época"),
        yaxis=_square_axis("Perda", log=True),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            orientation="v",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.update_yaxes(
        type="log",
        dtick=1,
        exponentformat="power",
        showexponent="all",
    )

    fig.show()


# ── 2. Mapas de calor ─────────────────────────────────────────────────────────

def plot_heatmaps(U_pred, U_ref, x, y, title='Solução'):
    """
    Plota mapas de calor: solução predita, referência e erro absoluto.

    Args:
        U_pred: array (n_grid, n_grid) — solução predita
        U_ref:  array (n_grid, n_grid) — solução de referência
        x:      array (n_grid,)        — coordenadas x
        y:      array (n_grid,)        — coordenadas y
        title:  título do plot
    """
    E_abs = np.abs(U_pred - U_ref)

    vmin = min(U_pred.min(), U_ref.min())
    vmax = max(U_pred.max(), U_ref.max())

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Predição", "Referência", "Erro absoluto"],
        horizontal_spacing=0.15,
        column_widths=[0.33, 0.33, 0.33],
    )

    colorbar_configs = [
        None,
        dict(x=0.655, y=0.5, len=0.9, thickness=12),
        dict(x=1.00,  y=0.5, len=0.9, thickness=12),
    ]

    for col, (Z, zmin, zmax, showscale, cbar) in enumerate(zip(
        [U_pred, U_ref,  E_abs],
        [vmin,   vmin,   0],
        [vmax,   vmax,   E_abs.max()],
        [False,  True,   True],
        colorbar_configs
    ), start=1):

        fig.add_trace(go.Heatmap(
            z=Z, x=x, y=y,
            colorscale=COLORSCALE,
            zmin=zmin, zmax=zmax,
            showscale=showscale,
            colorbar=cbar,
        ), row=1, col=col)

    for col in [1, 2, 3]:
        fig.update_xaxes(
            **AXIS_COMMON,
            title_text="x",
            title_font=dict(size=FONT_SIZE),
            range=[0, 1],
            constrain="domain",
            row=1, col=col,
        )
        fig.update_yaxes(
            **AXIS_COMMON,
            title_text="y" if col == 1 else "",
            title_font=dict(size=FONT_SIZE),
            range=[0, 1],
            constrain="domain",
            scaleanchor=f"x{'' if col == 1 else col}",
            scaleratio=1,
            row=1, col=col,
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        height=450,
        width=1000,
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


# ── 3. Perfis 1D ──────────────────────────────────────────────────────────────

def plot_profiles(U_pred_slices, U_ref_slices, x, slices=None, title='Perfis 1D'):
    """
    Plota perfis 1D comparando predição vs referência.

    Args:
        U_pred_slices: lista de arrays (n,) — predição em cada corte
        U_ref_slices:  lista de arrays (n,) — referência em cada corte
        x:             array (n,)           — coordenadas x
        slices:        lista de valores do corte (ex: [0.25, 0.5, 0.75])
        title:         título do plot
    """
    if slices is None:
        slices = [0.25, 0.5, 0.75]

    n = len(slices)

    rows, cols = (2, 2) if n == 4 else (1, n)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"y = {y_val}" for y_val in slices],
        shared_yaxes=True,
        horizontal_spacing=0.10,
        vertical_spacing=0.18,
    )

    for idx, (y_val, U_pred, U_ref) in enumerate(zip(slices, U_pred_slices, U_ref_slices)):
        row = idx // cols + 1
        col = idx %  cols + 1
        color = COLORS[idx % len(COLORS)]

        fig.add_trace(go.Scatter(
            x=x, y=U_pred,
            mode="lines",
            name="Predição",
            legendgroup="pred",
            line=dict(color=color, width=2.5),
            showlegend=(idx == 0),
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=x, y=U_ref,
            mode="lines",
            name="Referência",
            legendgroup="ref",
            line=dict(color=color, width=2.5, dash="dash"),
            showlegend=(idx == 0),
        ), row=row, col=col)

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(
                **AXIS_COMMON,
                title_text="x",
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )
            fig.update_yaxes(
                **AXIS_COMMON,
                title_text="u(x, y)" if c == 1 else "",
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )

    fig_w = SQ_SIZE + 80       if n == 4 else cols * (SQ_SIZE // 2) + 80
    fig_h = SQ_SIZE + 80       if n == 4 else SQ_SIZE // 2 + 120

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        width=fig_w,
        height=fig_h,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            orientation="v",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=FONT_SIZE)

    fig.show()


# ── 4. Distribuição de pontos ─────────────────────────────────────────────────

def plot_points(X_col, X_bc=None, title='Distribuição dos pontos de amostragem'):
    """
    Plota a distribuição dos pontos de colocação e, opcionalmente, de contorno.
    X_bc é opcional pois a hard-PINN não usa training points.

    Args:
        X_col: array (N_c, 2) — pontos de colocação
        X_bc:  array (N_b, 2) — pontos de contorno (opcional)
        title: título do plot
    """

    # converte para numpy se necessário
    if hasattr(X_col, 'detach'):
        X_col = X_col.detach().cpu().numpy()
    if X_bc is not None and hasattr(X_bc, 'detach'):
        X_bc = X_bc.detach().cpu().numpy()
        
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_col[:, 0], y=X_col[:, 1],
        mode="markers", name="Colocação",
        marker=dict(color="#2980B9", size=5, opacity=0.6),
    ))

    if X_bc is not None:
        fig.add_trace(go.Scatter(
            x=X_bc[:, 0], y=X_bc[:, 1],
            mode="markers", name="Contorno",
            marker=dict(color="#E74C3C", size=7, symbol="square"),
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=dict(
            **AXIS_COMMON,
            title=dict(text="x", font=dict(size=FONT_SIZE)),
            range=[-0.05, 1.05],
        ),
        yaxis=dict(
            **AXIS_COMMON,
            title=dict(text="y", font=dict(size=FONT_SIZE)),
            range=[-0.05, 1.05],
            scaleanchor="x",
            scaleratio=1,
        ),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            orientation="v",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()