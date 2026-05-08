"""
plot_utils.py — funções de visualização para notebooks de PINNs.

Todas as funções recebem arrays numpy prontos — nenhuma dependência de torch,
modelos ou funções analíticas. A conversão de tensores para arrays é
responsabilidade do script particular de cada exemplo.

Compatível com problemas estacionários (ex: Laplace) e transientes (ex: Burgers).
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
    Aceita qualquer combinação de: 'loss', 'loss_data', 'loss_pde', 'loss_ic'
 
    Args:
        history: dicionário com qualquer subconjunto de chaves:
                 'loss', 'loss_data', 'loss_pde', 'loss_ic'
    """
    labels = {
        'loss':      ('Total',    '#2C3E50', 'solid'),
        'loss_pde':  ('PDE',      '#2980B9', 'dot'),
        'loss_bc':   ('C. Cont.', '#E74C3C', 'dash'),      # condição de contorno
        'loss_ic':   ('C. Ini.',  '#27AE60', 'dashdot'),   # condição inicial (transiente)
        'loss_data': ('Dados',    '#8E44AD', 'longdash'),  # dados observados (inverso)
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

def plot_heatmaps(U_pred, U_ref, x, y, title='Solução', xlabel='x', ylabel='y',
                  square_aspect=None):
    """
    Plota mapas de calor: solução predita, referência e erro absoluto.

    Args:
        U_pred:        array (n_grid, n_grid) — solução predita
        U_ref:         array (n_grid, n_grid) — solução de referência
        x:             array (n_grid,)        — coordenadas x (1ª dimensão do grid)
        y:             array (n_grid,)        — coordenadas y ou t (2ª dimensão)
        title:         título do plot
        xlabel:        título do eixo x
        ylabel:        título do eixo y (use 't' para problemas transientes)
        square_aspect: força aspecto quadrado nos heatmaps (default: True se
                       domínio x e y têm o mesmo tamanho, False caso contrário)
    """
    E_abs = np.abs(U_pred - U_ref)

    vmin = min(U_pred.min(), U_ref.min())
    vmax = max(U_pred.max(), U_ref.max())

    # Infere aspecto automaticamente se não fornecido
    if square_aspect is None:
        x_range = x[-1] - x[0]
        y_range = y[-1] - y[0]
        square_aspect = np.isclose(x_range, y_range, rtol=0.05)

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

    x_range_plot = [x[0], x[-1]]
    y_range_plot = [y[0], y[-1]]

    for col in [1, 2, 3]:
        x_axis_kw = dict(
            **AXIS_COMMON,
            title_text=xlabel,
            title_font=dict(size=FONT_SIZE),
            range=x_range_plot,
            constrain="domain",
            row=1, col=col,
        )
        y_axis_kw = dict(
            **AXIS_COMMON,
            title_text=ylabel if col == 1 else "",
            title_font=dict(size=FONT_SIZE),
            range=y_range_plot,
            constrain="domain",
            row=1, col=col,
        )
        # Aspecto quadrado apenas quando domínios são compatíveis
        if square_aspect:
            y_axis_kw["scaleanchor"] = f"x{'' if col == 1 else col}"
            y_axis_kw["scaleratio"]  = 1

        fig.update_xaxes(**x_axis_kw)
        fig.update_yaxes(**y_axis_kw)

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

def plot_profiles(U_pred_slices, U_ref_slices, x, slices=None, title='Perfis 1D',
                  xlabel='x', ylabel='u(x)', slice_label='t', shared_yaxes=False):
    """
    Plota perfis 1D comparando predição vs referência.

    Args:
        U_pred_slices: lista de arrays (n,) — predição em cada corte
        U_ref_slices:  lista de arrays (n,) — referência em cada corte
        x:             array (n,)           — coordenadas x
        slices:        lista de valores do corte (ex: [0.25, 0.5, 0.75])
        title:         título do plot
        xlabel:        título do eixo x
        ylabel:        título do eixo y
        slice_label:   rótulo do parâmetro de corte ('t' ou 'y')
        shared_yaxes:  compartilha eixo y entre subplots (default: False)
                       Use True apenas para campos suaves com escalas similares.
                       Para Burgers ou choques, mantenha False.
    """
    if slices is None:
        slices = [0.25, 0.5, 0.75]

    n = len(slices)

    rows, cols = (2, 2) if n == 4 else (1, n)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{slice_label} = {val:.2f}" for val in slices],
        shared_yaxes=shared_yaxes,
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
                title_text=xlabel,
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )
            fig.update_yaxes(
                **AXIS_COMMON,
                title_text=ylabel if c == 1 else "",
                title_font=dict(size=FONT_SIZE),
                row=r, col=c,
            )

    fig_w = SQ_SIZE + 80 if n == 4 else cols * (SQ_SIZE // 2) + 80
    fig_h = SQ_SIZE + 80 if n == 4 else SQ_SIZE // 2 + 120

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

def plot_points_stationary(
    X_col, X_bc=None,
    title='Distribuição dos pontos (estacionário)',
    xlabel='x', ylabel='y',
    square_aspect=None
):
    def _to_np(arr):
        if arr is None:
            return None
        return arr.detach().cpu().numpy()

    X_col = _to_np(X_col)
    X_bc  = _to_np(X_bc)

    # Range automático
    all_pts = [X_col]
    if X_bc is not None:
        all_pts.append(X_bc)
    all_pts = np.vstack(all_pts)

    pad = 0.05
    x_range = [all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad]
    y_range = [all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad]

    # Aspecto automático
    if square_aspect is None:
        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]
        square_aspect = np.isclose(x_size, y_size, rtol=0.05)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_col[:, 0], y=X_col[:, 1],
        mode="markers",
        name="Colocação",
        marker=dict(size=5, opacity=0.6),
    ))

    if X_bc is not None:
        fig.add_trace(go.Scatter(
            x=X_bc[:, 0], y=X_bc[:, 1],
            mode="markers",
            name="Contorno",
            marker=dict(size=7, symbol="square"),
        ))

    yaxis_kw = dict(
        **AXIS_COMMON,
        title=dict(text=ylabel, font=dict(size=FONT_SIZE)),
        range=y_range,
    )

    if square_aspect:
        yaxis_kw["scaleanchor"] = "x"
        yaxis_kw["scaleratio"] = 1

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=dict(
            **AXIS_COMMON,
            title=dict(text=xlabel, font=dict(size=FONT_SIZE)),
            range=x_range,
        ),
        yaxis=yaxis_kw,
        height=450,
        width=700,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()

def plot_points_transient(
    X_col, X_bc=None, X_ic=None,
    title='Distribuição dos pontos (transiente)',
    xlabel='t', ylabel='x',
    square_aspect=None
):
    def _to_np(arr):
        if arr is None:
            return None
        return arr.detach().cpu().numpy()

    X_col = _to_np(X_col)
    X_bc  = _to_np(X_bc)
    X_ic  = _to_np(X_ic)

    # Range automático
    all_pts = [X_col]
    if X_bc is not None:
        all_pts.append(X_bc)
    if X_ic is not None:
        all_pts.append(X_ic)
    all_pts = np.vstack(all_pts)

    pad = 0.05
    x_range = [all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad]
    y_range = [all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad]

    # Aspecto automático (igual ao heatmap)
    if square_aspect is None:
        x_size = x_range[1] - x_range[0]
        y_size = y_range[1] - y_range[0]
        square_aspect = np.isclose(x_size, y_size, rtol=0.05)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X_col[:, 1], y=X_col[:, 0],
        mode="markers",
        name="Colocação",
        marker=dict(size=5, opacity=0.6),
    ))

    if X_bc is not None:
        fig.add_trace(go.Scatter(
            x=X_bc[:, 1], y=X_bc[:, 0],
            mode="markers",
            name="Contorno",
            marker=dict(size=7, symbol="square"),
        ))

    if X_ic is not None:
        fig.add_trace(go.Scatter(
            x=X_ic[:, 1], y=X_ic[:, 0],
            mode="markers",
            name="Cond. inicial",
            marker=dict(size=7, symbol="diamond"),
        ))

    yaxis_kw = dict(
        **AXIS_COMMON,
        title=dict(text=ylabel, font=dict(size=FONT_SIZE)),
        range=y_range,
    )

    if square_aspect:
        yaxis_kw["scaleanchor"] = "x"
        yaxis_kw["scaleratio"] = 1

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=dict(
            **AXIS_COMMON,
            title=dict(text=xlabel, font=dict(size=FONT_SIZE)),
            range=x_range,
        ),
        yaxis=yaxis_kw,
        height=450,
        width=700,
        legend=dict(
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()

def plot_psi0_evolution(history, psi0_true, title='Evolução de ψ₀'):
    """
    Plota a evolução do potencial de superfície durante o treinamento.

    Args:
        history:   dicionário com 'psi0' — lista de valores por época
        psi0_true: valor verdadeiro de psi0
        title:     título do plot
    """
    epochs = list(range(len(history['psi0'])))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, y=history['psi0'],
        mode='lines', name='ψ₀ recuperado',
        line=dict(color='#2C3E50', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=[epochs[0], epochs[-1]],
        y=[psi0_true, psi0_true],
        mode='lines', name='ψ₀ verdadeiro',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, font=dict(size=FONT_SIZE + 2)),
        xaxis=_square_axis('Época'),
        yaxis=_square_axis('ψ₀'),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            orientation='v',
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()


def plot_pb(results, X_obs, Psi_obs, title='Equação de Poisson-Boltzmann'):
    """
    Plota a solução da PINN, a solução exata e a curva recuperada,
    junto com os dados de observação ruidosos.

    Args:
        results:  dicionário retornado por evaluate_pb
        X_obs:    tensor (N_obs, 1) — posições das observações
        Psi_obs:  tensor (N_obs, 1) — potencial ruidoso
        title:    título do plot
    """
    X_obs_np   = X_obs.detach().cpu().numpy().ravel()
    Psi_obs_np = Psi_obs.detach().cpu().numpy().ravel()

    fig = go.Figure()

    # dados ruidosos
    fig.add_trace(go.Scatter(
        x=X_obs_np, y=Psi_obs_np,
        mode='markers', name='Observações',
        marker=dict(color='#2C3E50', size=7, symbol='circle-open')
    ))

    # solução exata
    fig.add_trace(go.Scatter(
        x=results['x'], y=results['psi_exact'],
        mode='lines', name=f"Exata (ψ₀={results['psi0_true']:.3f})",
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))

    # curva recuperada
    fig.add_trace(go.Scatter(
        x=results['x'], y=results['psi_recovered'],
        mode='lines', name=f"Recuperada (ψ₀={results['psi0_pred']:.3f})",
        line=dict(color='#2980B9', width=2, dash='dot')
    ))

    # predição da rede
    fig.add_trace(go.Scatter(
        x=results['x'], y=results['psi_pred'],
        mode='lines', name='PINN',
        line=dict(color='#27AE60', width=2)
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text=f"{title}<br><sup>Erro em ψ₀: {results['error_pct']:.2f}% | "
                 f"Erro L2: {results['l2_error']:.2e}</sup>",
            font=dict(size=FONT_SIZE + 2)
        ),
        xaxis=_square_axis('x (comprimentos de Debye)'),
        yaxis=_square_axis('ψ(x)'),
        width=SQ_SIZE,
        height=SQ_SIZE,
        legend=dict(
            x=1.02, y=1,
            xanchor='left', yanchor='top',
            orientation='v',
            font=dict(size=FONT_SIZE - 1),
            borderwidth=1,
        ),
    )

    fig.show()