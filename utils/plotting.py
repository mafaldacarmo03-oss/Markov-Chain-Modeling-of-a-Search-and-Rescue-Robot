from __future__ import annotations

import os
import tempfile

import gymnasium as gym
import streamlit as st

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots



from utils.discrete import is_absorbing_state, is_hazard, is_victim


try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    Network = None
    PYVIS_AVAILABLE = False


def format_state(state: tuple[int, int]) -> str:
    return f"{state}"


def draw_grid(
    grid: np.ndarray,
    robot_pos: tuple[int, int],
    path: list[tuple[int, int]],
    terminated: bool,
) -> go.Figure:
    n_rows, n_cols = grid.shape
    fig = go.Figure()

    color_map = {
        ".": "white",
        "R": "white",
        "H": "red",
        "V": "green",
    }

    for r in range(n_rows):
        for c in range(n_cols):
            cell = grid[r, c]
            y = n_rows - 1 - r

            fig.add_shape(
                type="rect",
                x0=c,
                y0=y,
                x1=c + 1,
                y1=y + 1,
                line=dict(color="gray"),
                fillcolor=color_map[cell],
                layer="below",
            )

            if cell in ["H", "V"]:
                fig.add_annotation(
                    x=c + 0.5,
                    y=y + 0.5,
                    text=cell,
                    showarrow=False,
                    font=dict(size=22, color="black"),
                    xanchor="center",
                    yanchor="middle",
                )

    path_x = []
    path_y = []

    for r, c in path:
        path_x.append(c + 0.5)
        path_y.append(n_rows - 1 - r + 0.5)

    fig.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines+markers",
            name="Robot path",
            line=dict(width=3),
            marker=dict(size=7),
        )
    )

    rr, rc = robot_pos
    robot_x = rc + 0.5
    robot_y = n_rows - 1 - rr + 0.5

    robot_color = "blue"

    if terminated and is_hazard(grid, robot_pos):
        robot_color = "red"
    elif terminated and is_victim(grid, robot_pos):
        robot_color = "green"

    fig.add_trace(
        go.Scatter(
            x=[robot_x],
            y=[robot_y],
            mode="markers",
            marker=dict(size=46, color=robot_color),
            name="Robot",
        )
    )

    fig.add_annotation(
        x=robot_x,
        y=robot_y,
        text="<b>R</b>",
        showarrow=False,
        font=dict(size=16, color="white"),
        xanchor="center",
        yanchor="middle",
    )

    fig.update_layout(
        height=600 if n_rows == 8 else 500,
        xaxis=dict(
            range=[0, n_cols],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[0, n_rows],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        title="Frozen-Lake Rescue Grid",
        showlegend=True,
    )

    return fig


def draw_transition_diagram(
    pos: tuple[int, int],
    transition_probs: dict[tuple[int, int], float],
    grid: np.ndarray,
) -> go.Figure:
    fig = go.Figure()

    current_x = 0
    current_y = 0

    fig.add_trace(
        go.Scatter(
            x=[current_x],
            y=[current_y],
            mode="markers+text",
            marker=dict(size=60, color="blue"),
            text=[format_state(pos)],
            textposition="bottom center",
            name="Current state",
        )
    )

    next_states = list(transition_probs.keys())

    if len(next_states) == 1:
        y_positions = [0]
    else:
        y_positions = np.linspace(1.8, -1.8, len(next_states))

    for i, next_state in enumerate(next_states):
        prob = transition_probs[next_state]

        next_x = 3
        next_y = y_positions[i]

        if is_hazard(grid, next_state):
            node_color = "red"
        elif is_victim(grid, next_state):
            node_color = "green"
        elif next_state == pos:
            node_color = "orange"
        else:
            node_color = "gray"

        fig.add_trace(
            go.Scatter(
                x=[next_x],
                y=[next_y],
                mode="markers+text",
                marker=dict(size=55, color=node_color),
                text=[format_state(next_state)],
                textposition="bottom center",
                name=f"Next state {i + 1}",
            )
        )

        fig.add_annotation(
            x=next_x - 0.25,
            y=next_y,
            ax=current_x + 0.25,
            ay=current_y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            text=f"{prob:.2f}",
            font=dict(size=14),
            bgcolor="white",
        )

    fig.update_layout(
        title="Current One-Step Transition Diagram",
        height=420,
        showlegend=False,
        xaxis=dict(visible=False, range=[-1, 4]),
        yaxis=dict(visible=False, range=[-2.5, 2.5]),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def draw_full_transition_graph_pyvis(
    matrix: np.ndarray,
    states: list[tuple[int, int]],
    grid: np.ndarray,
    start_pos: tuple[int, int],
    edge_threshold: float = 0.01,
):
    if not PYVIS_AVAILABLE:
        return None

    net = Network(
        height="600px",
        width="100%",
        directed=True,
        notebook=False,
        bgcolor="#ffffff",
        font_color="black",
    )

    net.barnes_hut(
        gravity=-3500,
        central_gravity=0.15,
        spring_length=180,
        spring_strength=0.035,
        damping=0.85,
    )

    for i, state in enumerate(states):
        state_id = f"s{i}"
        cell = grid[state]

        title = (
            f"{state_id}\n"
            f"Position: {state}\n"
            f"Cell type: {cell}\n"
            f"Absorbing: {is_absorbing_state(grid, state)}"
        )

        if is_hazard(grid, state):
            color = "#ef4444"
            title += "\nType: Failure absorbing state"
        elif is_victim(grid, state):
            color = "#22c55e"
            title += "\nType: Success absorbing state"
        elif state == start_pos:
            color = "#3b82f6"
            title += "\nType: Start state"
        else:
            color = "#d1d5db"
            title += "\nType: Safe frozen state"

        net.add_node(
            state_id,
            label=state_id,
            title=title,
            color=color,
            shape="dot",
            size=24,
        )

    for i, _from_state in enumerate(states):
        for j, _to_state in enumerate(states):
            prob = matrix[i, j]

            if prob >= edge_threshold:
                from_id = f"s{i}"
                to_id = f"s{j}"

                if i == j:
                    edge_color = "#f59e0b"
                    edge_width = 2
                else:
                    edge_color = "#6b7280"
                    edge_width = 1 + 5 * prob

                net.add_edge(
                    from_id,
                    to_id,
                    label=f"{prob:.2f}",
                    title=f"P({to_id} | {from_id}) = {prob:.4f}",
                    arrows="to",
                    color=edge_color,
                    width=edge_width,
                    smooth={
                        "enabled": True,
                        "type": "dynamic",
                    },
                )

    net.set_options(
        """
        {
          "nodes": {
            "font": {
              "size": 18,
              "face": "arial",
              "color": "#111827"
            },
            "borderWidth": 2
          },
          "edges": {
            "font": {
              "size": 13,
              "align": "middle",
              "color": "#111827",
              "strokeWidth": 3,
              "strokeColor": "#ffffff"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.8
              }
            },
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 700
            },
            "barnesHut": {
              "gravitationalConstant": -3500,
              "centralGravity": 0.15,
              "springLength": 180,
              "springConstant": 0.035,
              "damping": 0.85
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          }
        }
        """
    )

    return net


def pyvis_to_html(net) -> str:
    """
    Convert a PyVis network into an HTML string.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        path = tmp_file.name

    net.save_graph(path)

    with open(path, "r", encoding="utf-8") as file:
        html = file.read()

    os.remove(path)

    return html

def grid_to_frozenlake_desc(grid):
    """
    Convert the custom rescue grid to a Gymnasium FrozenLake desc.

    Custom app:
        R = robot start
        V = victim / success state
        H = hazard / failure state
        . = safe cell

    FrozenLake:
        S = start
        G = goal
        H = hole
        F = frozen/safe
    """
    desc = []

    for row in grid:
        line = ""

        for cell in row:
            if cell == "R":
                line += "S"
            elif cell == "V":
                line += "G"
            elif cell == "H":
                line += "H"
            else:
                line += "F"

        desc.append(line)

    return desc


def position_to_state(pos, grid):
    """
    Convert a grid position (row, col) to a FrozenLake state index.
    """
    n_rows, n_cols = grid.shape
    r, c = pos

    return r * n_cols + c


def render_frozenlake_state(
    grid,
    robot_pos,
    caption="Gymnasium FrozenLake rendered frame",
    image_width=500,
):
    """
    Render the current rescue-grid state using Gymnasium FrozenLake.

    The map is created from the app's custom grid, but the visible agent position
    is forced to match the app's simulation state.
    """
    desc = grid_to_frozenlake_desc(grid)

    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=True,
        render_mode="rgb_array",
    )

    env.reset()

    state = position_to_state(robot_pos, grid)

    # Force Gymnasium's current state to equal our app's robot position.
    env.unwrapped.s = state

    frame = env.render()
    env.close()

    st.image(
        frame,
        caption=caption,
        width=image_width,
    )




def draw_transition_matrix_heatmap(
    matrix,
    states,
    current_state_idx=None,
    highlight_col_idx=None,
    start_state_idx=None,
    most_likely_idx=None,
    show_colorbar=True,
):
    labels = [f"s{i} {states[i]}" for i in range(len(states))]
    n = len(states)

    text_values = np.round(matrix, 2)

    fig = go.Figure(
    data=go.Heatmap(
        z=matrix,
        x=list(range(n)),
        y=list(range(n)),
        text=text_values,
        texttemplate="%{text:.2f}",
        textfont=dict(size=10),
        colorscale="Blues",
        zmin=0,
        zmax=max(1.0, float(np.max(matrix))),
        showscale=show_colorbar,
        colorbar=dict(title="Probability"),
        hovertemplate=(
            "From %{y}<br>"
            "To %{x}<br>"
            "Probability: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title="Next state",
        yaxis_title="Current state",
        height=650,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        tickangle=-45,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        autorange="reversed",
    )

    # Backward compatibility: if current_state_idx is used, treat it as start row
    if start_state_idx is None and current_state_idx is not None:
        start_state_idx = current_state_idx

    # Backward compatibility: if highlight_col_idx is used, treat it as most likely column
    if most_likely_idx is None and highlight_col_idx is not None:
        most_likely_idx = highlight_col_idx

    # Highlight start-state row in orange
    if start_state_idx is not None:
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=n - 0.5,
            y0=start_state_idx - 0.5,
            y1=start_state_idx + 0.5,
            line=dict(color="orange", width=4),
            fillcolor="rgba(255, 165, 0, 0.08)",
            layer="above",
        )

        fig.add_annotation(
            x=n - 0.5,
            y=start_state_idx,
            text=f"start row: s{start_state_idx}",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=14, color="orange"),
            bgcolor="white",
        )

    # Highlight most likely final-state column in red
    if most_likely_idx is not None:
        fig.add_shape(
            type="rect",
            x0=most_likely_idx - 0.5,
            x1=most_likely_idx + 0.5,
            y0=-0.5,
            y1=n - 0.5,
            line=dict(color="red", width=4),
            fillcolor="rgba(255, 0, 0, 0.08)",
            layer="above",
        )

        fig.add_annotation(
            x=most_likely_idx,
            y=-0.5,
            text=f"most likely: s{most_likely_idx}",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=14, color="red"),
            bgcolor="white",
        )

    return fig

def draw_state_probability_grid(
    grid,
    states,
    distribution,
    current_pos=None,
    start_pos=None,
):
    """
    Draw a grid heatmap showing pi_n, the probability of being in each state.

    Optionally highlights the current sampled robot position.
    """
    n_rows, n_cols = grid.shape

    probability_grid = np.zeros((n_rows, n_cols), dtype=float)
    text_grid = np.empty((n_rows, n_cols), dtype=object)

    for i, state in enumerate(states):
        r, c = state
        cell = grid[r, c]

        probability_grid[r, c] = distribution[i]

        label = f"s{i}<br>{distribution[i]:.3f}"

        if cell == "R":
            label += "<br>Start"
        elif cell == "V":
            label += "<br>Victim"
        elif cell == "H":
            label += "<br>Hazard"

        text_grid[r, c] = label

    fig = go.Figure(
        data=go.Heatmap(
            z=probability_grid,
            x=[str(c) for c in range(n_cols)],
            y=[str(r) for r in range(n_rows)],
            text=text_grid,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Probability"),
            hovertemplate=(
                "row=%{y}, col=%{x}<br>"
                "Probability=%{z:.4f}<extra></extra>"
            ),
        )
    )

    # Highlight current sampled robot position
    if current_pos is not None:
        r, c = current_pos
        fig.add_shape(
            type="rect",
            x0=c - 0.5,
            x1=c + 0.5,
            y0=r - 0.5,
            y1=r + 0.5,
            line=dict(color="red", width=4),
            fillcolor="rgba(0,0,0,0)",
        )

        fig.add_annotation(
            x=c,
            y=r,
            text="🤖",
            showarrow=False,
            font=dict(size=24),
            xanchor="center",
            yanchor="middle",
        )

    # Highlight start position
    if start_pos is not None:
        r, c = start_pos
        fig.add_shape(
            type="rect",
            x0=c - 0.48,
            x1=c + 0.48,
            y0=r - 0.48,
            y1=r + 0.48,
            line=dict(color="green", width=2, dash="dot"),
            fillcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        width=450,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    fig.update_yaxes(
    autorange="reversed",
    showticklabels=False,
    title_text=None,
    )

    fig.update_xaxes(
        showticklabels=False,
        title_text=None,
    )

    return fig



def draw_absorbing_matrix_heatmap(P_reordered, labels, n_transient):
    fig = go.Figure(
        data=go.Heatmap(
            z=P_reordered,
            x=list(range(len(labels))),
            y=list(range(len(labels))),
            text=np.round(P_reordered, 2),
            texttemplate="%{text:.2f}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Probability"),
        )
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=labels,
        tickangle=-45,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(labels))),
        ticktext=labels,
        autorange="reversed",
    )

    # vertical separator between transient and absorbing columns
    fig.add_shape(
        type="line",
        x0=n_transient - 0.5,
        x1=n_transient - 0.5,
        y0=-0.5,
        y1=len(labels) - 0.5,
        line=dict(color="red", width=3),
    )

    # horizontal separator between transient and absorbing rows
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(labels) - 0.5,
        y0=n_transient - 0.5,
        y1=n_transient - 0.5,
        line=dict(color="red", width=3),
    )

    fig.add_annotation(
        x=n_transient / 2,
        y=n_transient / 2,
        text="Q",
        showarrow=False,
        font=dict(size=24, color="black"),
        bgcolor="white",
    )

    fig.add_annotation(
        x=n_transient + (len(labels) - n_transient) / 2,
        y=n_transient / 2,
        text="R",
        showarrow=False,
        font=dict(size=24, color="black"),
        bgcolor="white",
    )

    fig.add_annotation(
        x=n_transient / 2,
        y=n_transient + (len(labels) - n_transient) / 2,
        text="0",
        showarrow=False,
        font=dict(size=24, color="black"),
        bgcolor="white",
    )

    fig.add_annotation(
        x=n_transient + (len(labels) - n_transient) / 2,
        y=n_transient + (len(labels) - n_transient) / 2,
        text="I",
        showarrow=False,
        font=dict(size=24, color="black"),
        bgcolor="white",
    )

    fig.update_layout(
        title="Reordered absorbing-chain transition matrix",
        height=700,
        margin=dict(l=40, r=40, t=70, b=80),
    )

    return fig


def draw_generator_matrix_heatmap(
    Q,
    states,
    highlight_diagonal=True,
):
    labels = [f"s{i} {states[i]}" for i in range(len(states))]
    n = len(states)

    max_abs_value = float(np.max(np.abs(Q)))

    fig = go.Figure(
        data=go.Heatmap(
            z=Q,
            x=list(range(n)),
            y=list(range(n)),
            text=np.round(Q, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=10),
            colorscale="RdBu",
            zmin=-max_abs_value,
            zmax=max_abs_value,
            zmid=0,
            colorbar=dict(title="Rate"),
            hovertemplate=(
                "From state: %{y}<br>"
                "To state: %{x}<br>"
                "q_ij: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        tickangle=-45,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        autorange="reversed",
    )

    fig.update_layout(
        xaxis_title="To state",
        yaxis_title="From state",
        height=700,
        margin=dict(l=40, r=40, t=70, b=90),
    )

    if highlight_diagonal:
        for i in range(n):
            fig.add_shape(
                type="rect",
                x0=i - 0.5,
                x1=i + 0.5,
                y0=i - 0.5,
                y1=i + 0.5,
                line=dict(color="black", width=3),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

        fig.add_annotation(
            x=n - 1,
            y=0,
            text="Diagonal: total rate of leaving each state",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=13, color="black"),
            bgcolor="white",
        )

    return fig


def draw_continuous_transition_matrix_heatmap(
    P_t,
    states,
    start_state_idx=None,
    most_likely_idx=None,
):
    """
    Draw a heatmap for the continuous-time transition matrix P(t) = e^{Qt}.

    Each row is a probability distribution:
    P_ij(t) = P(X(t) = s_j | X(0) = s_i)
    """
    labels = [f"s{i} {states[i]}" for i in range(len(states))]
    n = len(states)

    text_values = np.round(P_t, 3)

    fig = go.Figure(
        data=go.Heatmap(
            z=P_t,
            x=list(range(n)),
            y=list(range(n)),
            text=text_values,
            texttemplate="%{text:.3f}",
            textfont=dict(size=10),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Probability"),
            hovertemplate=(
                "From state: s%{y}<br>"
                "To state: s%{x}<br>"
                "P_ij(t) = %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        tickangle=-45,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        autorange="reversed",
    )

    fig.update_layout(
        xaxis_title="State at time t",
        yaxis_title="Initial state",
        height=650,
        margin=dict(l=40, r=40, t=70, b=90),
    )

    if start_state_idx is not None:
        fig.add_shape(
            type="rect",
            x0=-0.5,
            x1=n - 0.5,
            y0=start_state_idx - 0.5,
            y1=start_state_idx + 0.5,
            line=dict(color="orange", width=4),
            fillcolor="rgba(255, 165, 0, 0.08)",
            layer="above",
        )

        fig.add_annotation(
            x=n - 0.5,
            y=start_state_idx,
            text=f"start row: s{start_state_idx}",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(size=13, color="orange"),
            bgcolor="white",
        )

    if most_likely_idx is not None:
        fig.add_shape(
            type="rect",
            x0=most_likely_idx - 0.5,
            x1=most_likely_idx + 0.5,
            y0=-0.5,
            y1=n - 0.5,
            line=dict(color="red", width=4),
            fillcolor="rgba(255, 0, 0, 0.08)",
            layer="above",
        )

        fig.add_annotation(
            x=most_likely_idx,
            y=-0.5,
            text=f"most likely: s{most_likely_idx}",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=13, color="red"),
            bgcolor="white",
        )

    return fig


def draw_probability_evolution_bar_animation(
    Q,
    states,
    start_index,
    transition_matrix_over_time,
    t_min=0.0,
    t_max=10.0,
    dt=0.1,
):
    """
    Animated bar plot showing the evolution of state probabilities over continuous time.

    The initial distribution is concentrated in start_index.
    For each time t, the distribution is:

        pi(t) = pi0 @ P(t)
        P(t) = exp(Qt)
    """

    times = np.round(np.arange(t_min, t_max + dt, dt), 3)
    n_states = len(states)

    # Initial distribution
    pi0 = np.zeros(n_states)
    pi0[start_index] = 1.0

    labels = [f"s{i}\n{states[i]}" for i in range(n_states)]

    distributions = []

    for t in times:
        P_t = transition_matrix_over_time(Q, t)
        pi_t = pi0 @ P_t
        distributions.append(pi_t)

    # Initial frame
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=distributions[0],
                text=np.round(distributions[0], 3),
                textposition="outside",
            )
        ]
    )

    # Animation frames
    frames = []

    for k, t in enumerate(times):
        frames.append(
            go.Frame(
                data=[
                    go.Bar(
                        x=labels,
                        y=distributions[k],
                        text=np.round(distributions[k], 3),
                        textposition="outside",
                    )
                ],
                name=str(t),
            )
        )

    fig.frames = frames

    # Slider steps
    slider_steps = []

    for t in times:
        slider_steps.append(
            {
                "method": "animate",
                "label": f"{t:.1f}",
                "args": [
                    [str(t)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 200, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.update_layout(
        title=f"Evolution of State Probabilities from Initial State s{start_index}",
        xaxis_title="State",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=600,
        margin=dict(l=40, r=40, t=80, b=120),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 200, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Time t = ",
                    "suffix": "",
                    "font": {"size": 16},
                },
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ],
    )

    fig.update_xaxes(tickangle=-45)

    return fig




def draw_continuous_process_animation(
    Q,
    states,
    grid,
    start_index,
    victim_index,
    hazard_indices,
    transition_matrix_over_time,
    t_min=0.0,
    t_max=10.0,
    dt=0.1,
):
    """
    Animated visualization of the continuous-time Markov chain.

    For each time t:
        P(t) = exp(Qt)
        pi(t) = pi0 P(t)

    The animation shows:
    1. Probability distribution on the rescue grid.
    2. Probability per state as a bar plot.
    3. Success, failure, and survival probabilities over time.
    """


    # ============================================================
    # Precompute probabilities over time
    # ============================================================

    times = np.round(np.arange(t_min, t_max + dt, dt), 3)

    n_states = len(states)
    n_rows, n_cols = grid.shape

    pi0 = np.zeros(n_states)
    pi0[start_index] = 1.0

    distributions = []
    success_probs = []
    failure_probs = []
    survival_probs = []

    for t in times:
        P_t = transition_matrix_over_time(Q, t)
        pi_t = pi0 @ P_t

        success_prob = pi_t[victim_index]
        failure_prob = sum(pi_t[i] for i in hazard_indices)
        survival_prob = 1.0 - success_prob - failure_prob

        distributions.append(pi_t)
        success_probs.append(success_prob)
        failure_probs.append(failure_prob)
        survival_probs.append(survival_prob)

    # ============================================================
    # Helper: convert probability vector into grid layout
    # ============================================================

    def distribution_to_grid(pi_t):
        probability_grid = np.zeros((n_rows, n_cols), dtype=float)
        text_grid = np.empty((n_rows, n_cols), dtype=object)

        for i, state in enumerate(states):
            r, c = state
            cell = grid[r, c]

            probability_grid[r, c] = pi_t[i]

            label = f"s{i}<br>{pi_t[i]:.3f}"

            if cell == "R":
                label += "<br>Start"
            elif cell == "V":
                label += "<br>Victim"
            elif cell == "H":
                label += "<br>Hazard"

            text_grid[r, c] = label

        return probability_grid, text_grid

    state_labels = [f"s{i}" for i in range(n_states)]

    grid_z0, grid_text0 = distribution_to_grid(distributions[0])

    # ============================================================
    # Figure layout
    # Top-left: smaller grid
    # Top-right: larger bar plot
    # Bottom: full-width absorption/survival plot
    # ============================================================

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "heatmap"}, {"type": "bar"}],
            [{"type": "scatter", "colspan": 2}, None],
        ],
        subplot_titles=[
            "Probability distribution on grid",
            "Probability per state",
            "Absorption and survival probabilities",
        ],
        column_widths=[0.32, 0.68],
        row_heights=[0.48, 0.52],
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    # ============================================================
    # Initial probability grid
    # ============================================================

    fig.add_trace(
        go.Heatmap(
            z=grid_z0,
            x=list(range(n_cols)),
            y=list(range(n_rows)),
            text=grid_text0,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(
                x=0.31,
                y=0.78,
                len=0.30,
                thickness=12,
            ),
            showscale=False,
            hovertemplate=(
                "row=%{y}, col=%{x}<br>"
                "Probability=%{z:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # ============================================================
    # Initial bar plot
    # ============================================================

    fig.add_trace(
        go.Bar(
            x=state_labels,
            y=distributions[0],
            text=np.round(distributions[0], 3),
            textposition="outside",
            name="State probability",
        ),
        row=1,
        col=2,
    )

    # ============================================================
    # Absorption and survival curves
    # ============================================================

    fig.add_trace(
        go.Scatter(
            x=times,
            y=success_probs,
            mode="lines",
            name="Success probability",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=failure_probs,
            mode="lines",
            name="Failure probability",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=survival_probs,
            mode="lines",
            name="Mission still on probability",
        ),
        row=2,
        col=1,
    )

    # Moving vertical time marker
    fig.add_trace(
        go.Scatter(
            x=[times[0], times[0]],
            y=[0, 1],
            mode="lines",
            name="Current time",
            line=dict(dash="dash", width=3),
        ),
        row=2,
        col=1,
    )

    # ============================================================
    # Animation frames
    # ============================================================

    frames = []

    for k, t in enumerate(times):
        grid_z, grid_text = distribution_to_grid(distributions[k])

        frames.append(
            go.Frame(
                name=str(t),
                data=[
                    go.Heatmap(
                        z=grid_z,
                        x=list(range(n_cols)),
                        y=list(range(n_rows)),
                        text=grid_text,
                        texttemplate="%{text}",
                        colorscale="Blues",
                        zmin=0,
                        zmax=1,
                        colorbar=dict(
                            x=0.31,
                            y=0.78,
                            len=0.30,
                            thickness=12,
                        ),
                        showscale=False
                    ),
                    go.Bar(
                        x=state_labels,
                        y=distributions[k],
                        text=np.round(distributions[k], 3),
                        textposition="outside",
                    ),
                    go.Scatter(
                        x=times,
                        y=success_probs,
                        mode="lines",
                        name="Success probability",
                    ),
                    go.Scatter(
                        x=times,
                        y=failure_probs,
                        mode="lines",
                        name="Failure probability",
                    ),
                    go.Scatter(
                        x=times,
                        y=survival_probs,
                        mode="lines",
                        name="Mission still on probability",
                    ),
                    go.Scatter(
                        x=[t, t],
                        y=[0, 1],
                        mode="lines",
                        name="Current time",
                        line=dict(dash="dash", width=3),
                    ),
                ],
                traces=[0, 1, 2, 3, 4, 5],
            )
        )

    fig.frames = frames

    # ============================================================
    # Slider and animation buttons
    # ============================================================

    slider_steps = []

    for t in times:
        slider_steps.append(
            {
                "method": "animate",
                "label": f"{t:.1f}",
                "args": [
                    [str(t)],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 300, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.update_layout(
        title=dict(
            text="Continuous-Time Evolution of the Rescue Mission",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
        ),
        height=850,
        margin=dict(l=40, r=40, t=170, b=90),
        showlegend=True,

        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,

                # Position above the subplots, below the title
                "x": 0.0,
                "y": 1.13,
                "xanchor": "left",
                "yanchor": "top",

                "direction": "right",
                "pad": {"r": 10, "t": 0},

                "bgcolor": "#e8f1ff",
                "bordercolor": "#2563eb",
                "borderwidth": 2,
                "font": {
                    "color": "#1f2937",
                    "size": 13,
                },

                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 300, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],

        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Time t = ",
                    "font": {"size": 16},
                },
                "pad": {"t": 50},
                "steps": slider_steps,
            }
        ],
    )
    # ============================================================
    # Axes formatting
    # ============================================================

    fig.update_xaxes(
    showticklabels=False,
    title_text=None,
    row=1,
    col=1,
    )

    fig.update_yaxes(
        showticklabels=False,
        title_text=None,
        autorange="reversed",
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="State",
        tickangle=-45,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Probability",
        range=[0, 1],
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="Time t",
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Probability",
        range=[0, 1],
        row=2,
        col=1,
    )

    return fig


def draw_fundamental_matrix_heatmap(N, transient_indices, states):
    """
    Heatmap for the fundamental matrix N = (I - Q)^(-1).

    N_ij is the expected number of visits to transient state j
    before absorption, starting from transient state i.
    """
    labels = [
        f"s{idx} {states[idx]}"
        for idx in transient_indices
    ]

    n = len(transient_indices)

    text_values = np.round(N, 2)

    fig = go.Figure(
        data=go.Heatmap(
            z=N,
            x=list(range(n)),
            y=list(range(n)),
            text=text_values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=10),
            colorscale="Blues",
            zmin=0,
            zmax=float(np.max(N)),
            colorbar=dict(title="Expected visits"),
            hovertemplate=(
                "Start transient state: %{y}<br>"
                "Visited transient state: %{x}<br>"
                "Expected visits: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(
        title_text="Visited transient state",
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        tickangle=-45,
    )

    fig.update_yaxes(
        title_text="Starting transient state",
        tickmode="array",
        tickvals=list(range(n)),
        ticktext=labels,
        autorange="reversed",
    )

    fig.update_layout(
        title="Fundamental Matrix N",
        height=650,
        margin=dict(l=40, r=40, t=70, b=90),
    )

    return fig

def draw_expected_steps_grid_heatmap(
    grid,
    states,
    transient_indices,
    expected_steps,
    start_pos=None,
):
    """
    Draw a grid heatmap showing the expected number of steps until absorption.

    Only transient states have expected-step values.
    Absorbing states are shown as 0 because absorption has already occurred.
    """
    n_rows, n_cols = grid.shape

    expected_grid = np.zeros((n_rows, n_cols), dtype=float)
    text_grid = np.empty((n_rows, n_cols), dtype=object)

    transient_to_value = {
        global_idx: expected_steps[local_idx]
        for local_idx, global_idx in enumerate(transient_indices)
    }

    for i, state in enumerate(states):
        r, c = state
        cell = grid[r, c]

        value = transient_to_value.get(i, 0.0)
        expected_grid[r, c] = value

        label = f"s{i}<br>{value:.2f}"

        if cell == "R":
            label += "<br>Start"
        elif cell == "V":
            label += "<br>Victim"
        elif cell == "H":
            label += "<br>Hazard"

        text_grid[r, c] = label

    max_value = float(np.max(expected_grid))

    if max_value == 0:
        max_value = 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=expected_grid,
            x=list(range(n_cols)),
            y=list(range(n_rows)),
            text=text_grid,
            texttemplate="%{text}",
            colorscale="Blues",
            zmin=0,
            zmax=max_value,
            colorbar=dict(title="Expected steps"),
            hovertemplate=(
                "row=%{y}, col=%{x}<br>"
                "Expected steps=%{z:.4f}<extra></extra>"
            ),
        )
    )

    if start_pos is not None:
        r, c = start_pos

        fig.add_shape(
            type="rect",
            x0=c - 0.5,
            x1=c + 0.5,
            y0=r - 0.5,
            y1=r + 0.5,
            line=dict(color="green", width=3, dash="dot"),
            fillcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        title="Expected Steps Until Absorption",
        width=500,
        height=450,
        margin=dict(l=40, r=40, t=70, b=40),
    )

    fig.update_xaxes(
        title_text="Column",
        tickmode="array",
        tickvals=list(range(n_cols)),
        ticktext=[str(c) for c in range(n_cols)],
    )

    fig.update_yaxes(
        title_text="Row",
        tickmode="array",
        tickvals=list(range(n_rows)),
        ticktext=[str(r) for r in range(n_rows)],
        autorange="reversed",
    )

    return fig


def draw_absorption_probabilities_heatmap(
    B,
    transient_indices,
    absorbing_indices,
    states,
    grid,
):
    """
    Heatmap for the absorption probability matrix B = N R.

    B_ij is the probability of being absorbed in absorbing state j,
    starting from transient state i.
    """
    row_labels = [
        f"s{idx} {states[idx]}"
        for idx in transient_indices
    ]

    col_labels = []

    for idx in absorbing_indices:
        state = states[idx]

        if is_victim(grid, state):
            col_labels.append(f"s{idx} Success")
        elif is_hazard(grid, state):
            col_labels.append(f"s{idx} Failure")
        else:
            col_labels.append(f"s{idx}")

    text_values = np.round(B, 3)

    fig = go.Figure(
        data=go.Heatmap(
            z=B,
            x=list(range(len(absorbing_indices))),
            y=list(range(len(transient_indices))),
            text=text_values,
            texttemplate="%{text:.3f}",
            textfont=dict(size=11),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Probability"),
            hovertemplate=(
                "Starting transient state: %{y}<br>"
                "Absorbing state: %{x}<br>"
                "Absorption probability: %{z:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(
        title_text="Absorbing state",
        tickmode="array",
        tickvals=list(range(len(absorbing_indices))),
        ticktext=col_labels,
        tickangle=-30,
    )

    fig.update_yaxes(
        title_text="Starting transient state",
        tickmode="array",
        tickvals=list(range(len(transient_indices))),
        ticktext=row_labels,
        autorange="reversed",
    )

    fig.update_layout(
        title="Absorption Probabilities",
        height=600,
        margin=dict(l=40, r=40, t=70, b=90),
    )

    return fig