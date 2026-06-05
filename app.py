from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from textwrap import dedent

import numpy as np
import plotly.graph_objects as go

import time
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.discrete import (
    build_transition_matrix,
    generate_random_grid,
    get_grid_positions,
    get_transition_distribution,
    is_absorbing_state,
    is_hazard,
    is_victim,
    matrix_to_dataframe,
    reset_simulation_state,
    state_table,
    step_robot,
    initial_state_distribution,
    state_distribution_after_n_steps,
    evolve_state_distribution,
    distribution_to_dataframe,
    classify_markov_chain,
    compute_success_before_hazard_probabilities,
    build_absorbing_reordered_matrix,
)


from utils.continuous import (
    build_generator_matrix_from_transition_matrix,
    transition_matrix_over_time,
)


from utils.plotting import (
    draw_full_transition_graph_pyvis,
    draw_transition_diagram,
    draw_transition_matrix_heatmap,
    draw_state_probability_grid,
    pyvis_to_html,
    render_frozenlake_state,
    draw_absorbing_matrix_heatmap,
    draw_generator_matrix_heatmap,
    draw_continuous_transition_matrix_heatmap,
    draw_probability_evolution_bar_animation,
    draw_continuous_process_animation,
    draw_fundamental_matrix_heatmap,
    draw_expected_steps_grid_heatmap,
    draw_absorption_probabilities_heatmap
)

from utils.policies import get_valid_actions


st.set_page_config(
    page_title="Rescue Robot Markov Chains",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Markov Chain Modeling of a Search-and-Rescue Robot")
st.caption("From discrete-time navigation to continuous-time mission dynamics")


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.title("Model Controls")

grid_size_option = st.sidebar.selectbox(
    "Grid size",
    ["4x4", "8x8"],
)

grid_size = int(grid_size_option.split("x")[0])

default_hazards = 0 if grid_size == 4 else 0
max_hazards = grid_size * grid_size - 2

n_hazards = st.sidebar.slider(
    "Number of holes",
    min_value=0,
    max_value=max_hazards,
    value=min(default_hazards, max_hazards),
    step=1,
)

policy_type = st.sidebar.selectbox(
    "Policy",
    ["Random", "Greedy"],
)

success_prob = st.sidebar.slider(
    "Probability of intended movement",
    min_value=0.25,
    max_value=1.0,
    value=1.0,
    step=0.05,
)

edge_threshold = st.sidebar.slider(
    "Minimum edge probability shown",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01,
)

transition_rate = st.sidebar.slider(
    "Continuous-time transition rate λ",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
)

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=1,
    step=1,
)

generate_new_grid = st.sidebar.button("Generate New Grid")


# ============================================================
# Grid/session state management
# ============================================================

# Keep the seed actually used by the current grid in session_state.
# This avoids losing the generated seed after Streamlit reruns.
if "active_grid_seed" not in st.session_state:
    st.session_state.active_grid_seed = int(random_seed)

if "last_sidebar_seed" not in st.session_state:
    st.session_state.last_sidebar_seed = int(random_seed)

# If the user manually changes the sidebar seed, use that as the active seed.
if int(random_seed) != st.session_state.last_sidebar_seed:
    st.session_state.active_grid_seed = int(random_seed)
    st.session_state.last_sidebar_seed = int(random_seed)

# If the user clicks Generate New Grid, create and store a new active seed.
if generate_new_grid:
    st.session_state.active_grid_seed = random.randint(0, 1_000_000)
    st.session_state.last_sidebar_seed = int(random_seed)

grid_config = {
    "grid_size": grid_size,
    "n_hazards": n_hazards,
    "seed": st.session_state.active_grid_seed,
}

if "grid_config" not in st.session_state:
    st.session_state.grid_config = None

if "grid" not in st.session_state:
    st.session_state.grid = None

should_generate_grid = (
    st.session_state.grid is None
    or st.session_state.grid_config != grid_config
)

if should_generate_grid:
    try:
        st.session_state.grid = generate_random_grid(
            size=grid_size,
            n_hazards=n_hazards,
            seed=st.session_state.active_grid_seed,
        )

        st.session_state.grid_config = grid_config.copy()

    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    # Important: whenever a new grid is generated,
    # reset all robot simulation variables.
    start_pos, _victim_pos = get_grid_positions(st.session_state.grid)
    reset_simulation_state(st.session_state, start_pos)

    # Optional but useful: clear any extra cached/derived values
    # from previous simulations if you add them later.
    st.session_state.last_transition = None


grid = st.session_state.grid
n_rows, n_cols = grid.shape
start_pos, victim_pos = get_grid_positions(grid)

# Safety check: if robot_pos somehow does not exist, reset it.
if "robot_pos" not in st.session_state:
    reset_simulation_state(st.session_state, start_pos)


# ============================================================
# Main tabs
# ============================================================


tab_overview, tab_discrete, tab_continuous  = st.tabs(
    [
        "0. Overview",
        "1. Discrete-Time Markov Chain",
        "2. Continuous-Time Markov Chain",
       
    ]
)

# ============================================================
# Tab 0 - Overview
# ============================================================

with tab_overview:
    st.header("Project overview")
    st.write(
        """
        This app illustrates a simplified search-and-rescue robot operating in a hazardous
        grid environment. The same mission is studied in two ways:

        1. A **discrete-time Markov chain**, where the robot moves step by step on a grid.
        2. A **continuous-time Markov chain**, where the robot spends random amounts of
           real time in mission states before transitioning.
        """
    )

    st.subheader("Rescue interpretation of FrozenLake")

    col1,col2=st.columns(2)
    with col1:

        st.subheader("Gymnasium rescue environment")
        render_frozenlake_state(
            grid=grid,
            robot_pos=st.session_state.robot_pos,
            caption="FrozenLake rendered",
            image_width=400,
        )

    with col2: 
        mapping = pd.DataFrame(
            {
                "FrozenLake object": [
                    "Agent",
                    "Start ",
                    "Frozen tile ",
                    "Hole ",
                    "Goal ",
                    "Slippery movement",
                ],
                "Rescue-robot interpretation": [
                    "Search-and-rescue robot",
                    "Deployment point / base",
                    "Safe traversable terrain",
                    "Hazard, rubble, unstable floor, mission failure",
                    "Victim / rescue target / beacon",
                    "Motion uncertainty, wheel slip, sensor error",
                ],
            }
        )
        st.dataframe(mapping, hide_index=True, use_container_width=True)

    


# ============================================================
# Tab 1: Discrete-time Markov chain
# ============================================================

with tab_discrete:
    

    P, states = build_transition_matrix(
        grid=grid,
        policy_type=policy_type,
        success_prob=success_prob,
        victim_pos=victim_pos,
    )

    col1, col2 = st.columns([3, 1])

    with col1: 
        st.subheader("Discrete-Time Markov Chain Approach")

        st.markdown(
        """
        The robot moves on a frozen-lake-style rescue grid with uncertain motion. The state is only the robot position:  
        $$ 
        s_t = \\text{robot position at time } t 
        $$

        At each step, the robot first chooses an **intended movement direction** according to the selected policy:

        $$
        a_t \\in \\{\\text{UP},\\text{DOWN},\\text{LEFT},\\text{RIGHT}\\}.
        $$

        The policy determines the probability of intending each action:  
        $$ 
        \\pi(a \\mid s)=P(\\text{intended action}=a \\mid \\text{current state}=s). 
        $$

        For example, under the **Random** policy, if the robot is in a boundary cell where only
        **UP**, **DOWN**, and **RIGHT** are valid intended actions, then:

        $$
        \\pi(\\text{UP}\\mid s)=\\pi(\\text{DOWN}\\mid s)=\\pi(\\text{RIGHT}\\mid s)=\\frac{1}{3}.
        $$

        After the intended action is chosen, the terrain uncertainty is applied. If the probability
        of correctly executing the intended movement is \(p\), then:  
        $$ 
        P(\\text{actual action}=\\text{intended action})=p 
        $$

        and the remaining probability is distributed among the other three directions:  
        $$ 
        P(\\text{actual action}=\\text{each other direction})  = \\frac{1-p}{3}. 
        $$

        Therefore, the final transition probabilities \(P(s'|s)\) combine both the policy probabilities and the movement uncertainty.

        If a movement would take the robot outside the grid, the robot remains in the same cell.
        This creates a self-transition probability.

        Every grid cell is a Markov state. For an $$(n \\times n)$$ grid, the **transition matrix** has size:  $$  n^2 \\times n^2 $$

        The victim cell **V** is the absorbing success state, and each hole **H** is an absorbing failure state. 
        Therefore: $$  P(V \\mid V)=1 $$  and $$ P(H \\mid H)=1. $$

        ----
        """
    )
   
    with col2: 
        st.subheader("Model Summary")
        st.write(f"Grid size: `{grid_size_option}`")
        st.write(f"Number of Markov states: `{len(states)}`")
        st.write(f"Expected number of states: `{grid_size * grid_size}`")
        st.write(f"Number of holes: `{n_hazards}`")
        st.write(f"Policy: `{policy_type}`")
        st.write(f"Probability of intended movement: `{success_prob}`")
        st.write(f"Start position: `{start_pos}`")
        st.write(f"Victim/success position: `{victim_pos}`")
        st.write(
            f"Absorbing states: "
            f"`{sum(is_absorbing_state(grid, s) for s in states)}`"
        )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.subheader("Gymnasium rescue environment")
        render_frozenlake_state(
            grid=grid,
            robot_pos=st.session_state.robot_pos,
            caption="FrozenLake rendered",
            image_width=400,
        )

    with col_b:
        st.subheader("Transition Diagram")

        full_graph = draw_full_transition_graph_pyvis(
            matrix=P,
            states=states,
            grid=grid,
            start_pos=start_pos,
            edge_threshold=edge_threshold,
        )

        if full_graph is None:
            st.warning("PyVis is not installed. Run: `pip install pyvis`")
        else:
            if grid_size == 8:
                st.info(
                    "For 8x8 grids, the full transition diagram can become crowded. "
                    "Increase the edge-probability threshold in the sidebar if needed."
                )

            html = pyvis_to_html(full_graph)
            components.html(html, height=650, scrolling=True)

    

    st.markdown("---")


    st.subheader("Classification of the Markov Chain (Tarjan alg. - strongly connected components)")

    classification_df = classify_markov_chain(
        P=P,
        states=states,
        grid=grid,
    )

    classification_display = classification_df.copy()

    if "Cell types" in classification_display.columns:
        classification_display = classification_display.drop(columns=["Cell types"])

    classification_display["Closed"] = classification_display["Closed"].map({
        True: "✓",
        False: "—",
    })

    classification_display["Absorbing"] = classification_display["Absorbing"].map({
        True: "✓",
        False: "—",
    })

    def row_style(row):
        recurrence = row["Recurrent / transient"]
        absorbing = row["Absorbing"]

        if absorbing == "✓":
            return "background-color: #e8f1ff;"
        elif recurrence == "recurrent":
            return "background-color: #e9fbe9;"
        elif recurrence == "transient":
            return "background-color: #fff4d6;"
        return ""


    html = """
    <style>
    .markov-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 20px;
        line-height: 1.45;
    }

    .markov-table th {
        background-color: #f3f4f6;
        font-size: 20px;
        font-weight: 700;
        text-align: left;
        padding: 12px;
        border: 1px solid #d1d5db;
    }

    .markov-table td {
        padding: 12px;
        border: 1px solid #d1d5db;
        vertical-align: top;
    }

    .markov-table .class-col {
        font-weight: 700;
    }

    .markov-table .recurrent {
        color: #15803d;
        font-weight: 700;
    }

    .markov-table .transient {
        color: #b45309;
        font-weight: 700;
    }
    </style>

    <table class="markov-table">
    <thead>
    <tr>
    """

    for col in classification_display.columns:
        html += f"<th>{col}</th>"

    html += """
    </tr>
    </thead>
    <tbody>
    """

    for _, row in classification_display.iterrows():
        bg_style = row_style(row)
        html += f'<tr style="{bg_style}">'

        for col in classification_display.columns:
            value = row[col]

            if col == "Class":
                html += f'<td class="class-col">{value}</td>'
            elif col == "Recurrent / transient":
                css_class = "recurrent" if value == "recurrent" else "transient"
                html += f'<td class="{css_class}">{value}</td>'
            else:
                html += f"<td>{value}</td>"

        html += "</tr>"

    html += """
    </tbody>
    </table>
    """

    st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        """
        A communication class is a set of states that can reach each other.
        A class is **closed** if no transition leaves the class. In a finite-state Markov chain,
        closed classes are **recurrent**, while states outside closed classes are **transient**.

        The **period** of a class is the greatest common divisor of all possible return times.
        If the period is 1, the class is **aperiodic**.
        """
    )



    st.markdown("---")

    st.subheader("One Step Transition Probabilities")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("Run 1 Step", key="discrete_button_1_step"):
            step_robot(
                session_state=st.session_state,
                grid=grid,
                policy_type=policy_type,
                success_prob=success_prob,
                victim_pos=victim_pos,
            )

        current_pos = st.session_state.robot_pos
        current_state_idx = states.index(current_pos)

        if st.session_state.last_transition is not None:
            intended_action = st.session_state.last_transition["intended_action"]
            actual_action = st.session_state.last_transition["actual_action"]

            st.markdown (f"Intended action: {intended_action}   Actual action: {actual_action}")
        else:
            st.write("Intended action: None yet Actual action: None yet")

    with col2:
        if st.button("Reset", key="discrete_button_1_step_reset"):
            reset_simulation_state(st.session_state, start_pos)

        current_pos = st.session_state.robot_pos
        current_state_idx = states.index(current_pos)

        st.write(f"Current state: {current_state_idx} Current position: {current_pos}")
        st.write()


    col1, col2 = st.columns([1, 2])

    with col1:

        # This is the row of the transition matrix corresponding to the current state.
        # It gives P(s_j | current_state) for all possible next states s_j.
        one_step_distribution = P[current_state_idx, :]


        render_frozenlake_state(
            grid=grid,
            robot_pos=current_pos,
            caption=f"Current state: s{current_state_idx} | Position: {current_pos}",
            image_width=250,
        )

        st.markdown(
            f"##### One-step probabilities from state $s_{{{current_state_idx}}}$"
        )

        st.plotly_chart(
            draw_state_probability_grid(
                grid=grid,
                states=states,
                distribution=one_step_distribution,
                current_pos=current_pos,
                start_pos=start_pos,
            ),
            key=f"one_step_distribution_from_s{current_state_idx}",
        )

    with col2:
        st.markdown("##### Transition Matrix - One Step")

        st.markdown(
            """
            Each row represents the current state $s$, and each column represents the next state $s'$.
            The entry $P_{ij}$ is the probability of moving from state $s_i$ to state $s_j$ in one step.
            """
        )

        current_pos = st.session_state.robot_pos
        current_state_idx = states.index(current_pos)

        st.plotly_chart(
            draw_transition_matrix_heatmap(
                P,
                states,
                current_state_idx=current_state_idx,
            ),
            use_container_width=True,
            key=f"transition_matrix_heatmap_one_step_s{current_state_idx}",
        )


    st.markdown("---")

    st.markdown("---")

    st.subheader("Where is the robot likely to be after *n* movement steps?")

    col_steps, col_formula = st.columns([1, 4])

    with col_steps:
        n_steps = st.number_input(
            "Number of movement steps",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            format="%d",
            key="n_step_location_prediction",
        )

    n_steps = int(n_steps)

    with col_formula:
        st.markdown(
            f"""
            Starting from the initial state, the state distribution after **{n_steps}** steps is:

            $$
            \\pi_{{{n_steps}}} = \\pi_0 P^{{{n_steps}}}
            $$
            """
        )

    pi0 = np.zeros(len(states), dtype=float)
    start_index = states.index(start_pos)
    pi0[start_index] = 1.0

    P_n = np.linalg.matrix_power(P, n_steps)
    pi_n = pi0 @ P_n

    most_likely_idx = int(np.argmax(pi_n))
    most_likely_state = states[most_likely_idx]
    most_likely_prob = pi_n[most_likely_idx]



    def vector_box(values, decimals=3):
        if decimals == 0:
            vector_text = "[" + ", ".join(f"{int(x)}" for x in values) + "]"
        else:
            vector_text = "[" + ", ".join(f"{x:.{decimals}f}" for x in values) + "]"

        return f"""
        <div style="
            font-family: monospace;
            font-size: 12px;
            line-height: 1.6;
            background-color: #f8fafc;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 10px;
            overflow-x: auto;
            white-space: nowrap;
            margin-top: 95px;
        ">
            {vector_text}
        </div>
        """



    col_vec0, col_times, col_matrix, col_equals, col_vecn = st.columns(
        [1.2, 0.12, 3.4, 0.12, 1.5]
    )

    with col_vec0:
        st.markdown("##### Initial vector $\\pi_0$")

        st.markdown(
            vector_box(pi0, decimals=0),
            unsafe_allow_html=True,
        )

        st.caption(f"Start state: $s_{{{start_index}}}$")

    with col_times:
        st.markdown(
            """
            <div style="font-size: 42px; text-align: center; padding-top: 270px;">
                ×
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_matrix:
        st.markdown(f"##### Transition matrix $P^{{{n_steps}}}$")

        st.plotly_chart(
            draw_transition_matrix_heatmap(
                P_n,
                states,
                show_colorbar=False,
            ),
            use_container_width=True,
            key=f"calculation_transition_matrix_power_{n_steps}",
        )

    with col_equals:
        st.markdown(
            """
            <div style="font-size: 42px; text-align: center; padding-top: 270px;">
                =
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_vecn:
        st.markdown(f"##### Result $\\pi_{{{n_steps}}}$")

        st.markdown(
            vector_box(pi_n, decimals=3),
            unsafe_allow_html=True,
        )

        st.caption(
            f"Most likely state: $s_{{{most_likely_idx}}}$ "
            f"with probability {most_likely_prob:.4f}."
        )



    col_highlight_matrix, col_distribution_grid = st.columns([2, 1])

    with col_highlight_matrix:

        st.plotly_chart(
            draw_transition_matrix_heatmap(
                P_n,
                states,
                start_state_idx=start_index,
                most_likely_idx=most_likely_idx,
                show_colorbar=False,
            ),
            use_container_width=True,
            key=f"highlighted_transition_matrix_power_{n_steps}",
        )

    with col_distribution_grid:
        st.markdown(f"#### Distribution $\\pi_{{{n_steps}}}$")

        st.plotly_chart(
            draw_state_probability_grid(
                grid=grid,
                states=states,
                distribution=pi_n,
                current_pos=most_likely_state,
                start_pos=start_pos,
            ),
            use_container_width=True,
            key=f"pi_n_distribution_grid_{n_steps}",
        )


    st.markdown(
        f"""
        Given that the robot starts in state $s_{{{start_index}}}$, the relevant row of
        $P^{{{n_steps}}}$ is the **orange row**. The most likely final state is
        $s_{{{most_likely_idx}}}$, represented by the **red column**. Therefore, the
        probability of being in the most likely state after **{n_steps}** steps is the
        value at their intersection:

        $$
        (P^{{{n_steps}}})_{{s_{{{start_index}}},s_{{{most_likely_idx}}}}}
        =
        P(X_{{{n_steps}}}=s_{{{most_likely_idx}}}\\mid X_0=s_{{{start_index}}})
        =
        {most_likely_prob:.4f}.
        $$

        If the robot started from any other state, the probability of ending in
        $s_{{{most_likely_idx}}}$ after **{n_steps}** steps would be given by the corresponding 
        value in the same red column.
        """
    )

   
    st.markdown(f"#### One sampled trajectory up to {n_steps} steps")

    st.markdown(
        """
        The matrix calculation gives the full probability distribution after several steps.

        The animation below shows one sampled robot trajectory. At the same time,
        the probability grid shows how the full state distribution evolves after each step.
        """
    )

    with st.expander("Animate sampled robot trajectory and probability distribution", expanded=True):
        speed = 0.35

        if st.button(
            "Run trajectory animation",
            key=f"run_trajectory_animation_{n_steps}",
        ):
            

            # Temporary simulation state.
            # This avoids changing the main app session state.
            sim_state = SimpleNamespace()
            reset_simulation_state(sim_state, start_pos)

            # Initial distribution pi_0
            pi0_anim = np.zeros(len(states))
            pi0_anim[start_index] = 1.0

            trajectory_rows = []

            frame_col, prob_col = st.columns([1, 1])

            frame_placeholder = frame_col.empty()
            prob_placeholder = prob_col.empty()

            for step_idx in range(n_steps + 1):
                current_pos = sim_state.robot_pos
                current_state_idx = states.index(current_pos)

                # Full probability distribution after step_idx steps
                P_k = np.linalg.matrix_power(P, step_idx)
                pi_k = pi0_anim @ P_k

                most_likely_idx_k = int(np.argmax(pi_k))
                most_likely_state_k = states[most_likely_idx_k]

                trajectory_rows.append(
                    {
                        "Step": step_idx,
                        "Sampled State": f"s{current_state_idx}",
                        "Sampled Position": str(current_pos),
                        "Most Likely State": f"s{most_likely_idx_k}",
                        "Most Likely Position": str(most_likely_state_k),
                        "Most Likely Probability": pi_k[most_likely_idx_k],
                        "Type": (
                            "Victim"
                            if is_victim(grid, current_pos)
                            else "Hazard"
                            if is_hazard(grid, current_pos)
                            else "Start"
                            if current_pos == start_pos
                            else "Safe"
                        ),
                    }
                )

                with frame_placeholder.container():
                    st.markdown("##### Sampled robot trajectory")

                    render_frozenlake_state(
                        grid=grid,
                        robot_pos=current_pos,
                        caption=(
                            f"Step {step_idx} | "
                            f"Sampled state s{current_state_idx} | "
                            f"Position {current_pos}"
                        ),
                        image_width=300,
                    )

                with prob_placeholder.container():
                    st.markdown(f"##### Probability distribution $\\pi_{{{step_idx}}}$")

                    st.plotly_chart(
                        draw_state_probability_grid(
                            grid=grid,
                            states=states,
                            distribution=pi_k,
                            current_pos=current_pos,      # sampled robot position
                            start_pos=start_pos,
                        ),
                        use_container_width=True,
                        key=f"animated_probability_grid_step_{step_idx}_{n_steps}",
                    )

                    st.caption(
                        f"Sampled robot state: s{current_state_idx}. "
                        f"Most likely state after {step_idx} steps: "
                        f"s{most_likely_idx_k} with probability {pi_k[most_likely_idx_k]:.4f}."
                    )

                time.sleep(speed)

                if sim_state.terminated:
                    break

                if step_idx < n_steps:
                    step_robot(
                        session_state=sim_state,
                        grid=grid,
                        policy_type=policy_type,
                        success_prob=success_prob,
                        victim_pos=victim_pos,
                    )


    st.markdown("---")

    st.subheader("Absorving Markov Chain Analysis ")

     # Build the transition matrix
    P, states = build_transition_matrix(
        grid=grid,
        policy_type=policy_type,
        success_prob=success_prob,
        victim_pos=victim_pos,
    )
    
    # Identify absorbing and transient states
    absorbing_indices = []
    transient_indices = []
    
    for i, state in enumerate(states):
        if is_absorbing_state(grid, state):
            absorbing_indices.append(i)
        else:
            transient_indices.append(i)

    st.markdown(""" 
        ##### State Classification 
        In an absorbing Markov chain, the non-absorbing states are called **transient states**.
        These are states that the robot may visit temporarily before eventually reaching an  absorbing state.

        In this model, the absorbing states are:

         the victim state, which represents mission success;
         the hazard states, which represent mission failure.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Total states:** {len(states)}")
        st.write(f"**Transient states:** {len(transient_indices)}")
        st.write(f"**Absorbing states:** {len(absorbing_indices)}")
    
    # Show which states are absorbing
    absorbing_states_info = []
    for idx in absorbing_indices:
        state = states[idx]
        if is_victim(grid, state):
            status = "Victim (Success)"
        elif is_hazard(grid, state):
            status = "Hazard (Failure)"
        else:
            status = "Unknown"
        absorbing_states_info.append({
            "State": f"s{idx}",
            "Position": str(state),
            "Type": status
        })

    with col2:
        if absorbing_states_info:
            st.dataframe(
                pd.DataFrame(absorbing_states_info),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")

    st.markdown("##### Probability of reaching the target before entering a hazard")

    success_before_hazard = compute_success_before_hazard_probabilities(
        P=P,
        states=states,
        grid=grid,
        victim_pos=victim_pos,
    )

    start_idx = states.index(start_pos)
    victim_idx = states.index(victim_pos)

    hazard_indices = [
        i for i, state in enumerate(states)
        if is_hazard(grid, state)
    ]

    success_from_start = success_before_hazard[start_idx]
    failure_from_start = 1.0 - success_from_start

    st.markdown(
    f"""
    The **eventual probability**  of reaching the victim before entering any hazard.

    For each state $s_i$, define

    $$
    h_i =
    P(\\text{{reach victim before hazard at any future time}} \\mid X_0=s_i).
    $$

    Therefore, **$h_i$ is the long-run success probability when the robot starts from state $s_i$**.
    It considers all possible path lengths:

    $$
    1,2,3,\\ldots,\\infty
    $$

    until the Markov chain is absorbed either by the victim state or by a hazard state.

    The boundary conditions are immediate:

    $$
    h_i = 1
    \\quad \\text{{if }} s_i \\text{{ is the victim}},
    $$

    because if the robot is already at the victim, success is certain.

    $$
    h_i = 0
    \\quad \\text{{if }} s_i \\text{{ is a hazard}},
    $$

    because if the robot is already in a hazard, failure has already occurred.

    For every other state, we use the Markov property. From state $s_i$, the robot first moves
    to some next state $s_j$ with probability $P_{{ij}}$. From $s_j$, the process continues
    until success or failure. Therefore,

    $$
    h_i = \\sum_j P_{{ij}}h_j.
    $$

    This means that the success probability from state $s_i$ is a weighted average of the
    success probabilities of all states reachable in one step.

    For example, if from state $s_3$,

    $$
    P_{{3,4}}=0.6, \\quad P_{{3,7}}=0.3, \\quad P_{{3,3}}=0.1,
    $$

    then

    $$
    h_3 = 0.6h_4 + 0.3h_7 + 0.1h_3.
    $$

    In matrix form, we separate the states into transient states and absorbing states.
    After reordering, the transition matrix has the form

    $$
    P =
    \\begin{{pmatrix}}
    Q & R \\\\
    0 & I
    \\end{{pmatrix}},
    $$

    where $Q$ contains transitions among transient states, and $R$ contains transitions
    from transient states to absorbing states.

    For the success-before-hazard calculation, we only need the column of $R$
    corresponding to the victim. Call this column $r$. Then

    $$
    h_T = Qh_T + r.
    $$

    Moving terms to one side gives

    $$
    (I-Q)h_T = r.
    $$

    Therefore,

    $$
    h_T = (I-Q)^{{-1}}r.
    $$

    Starting from the initial state $s_{{{start_idx}}}=({start_pos[0]}, {start_pos[1]})$, the probability of
    eventually reaching the victim before entering a hazard is

    $$    h_{{{start_idx}}} = {success_from_start:.4f}.
    $$

    The corresponding probability of entering a hazard before reaching the victim is

    $$
    1-h_{{{start_idx}}} = {failure_from_start:.4f}.
    $$
    """
    )

    if len(hazard_indices) == 0:
        st.info(
            "There are no hazards in the current grid. "
            "Therefore, the probability of reaching the target before entering a hazard is trivial. "
            "Increase the number of holes to make this analysis meaningful."
        )



    st.markdown("#### Absorbing-Chain Matrix")

    absorbing_matrix_data = build_absorbing_reordered_matrix(
        P=P,
        states=states,
        grid=grid,
        victim_pos=victim_pos,
    )

    P_reordered = absorbing_matrix_data["P_reordered"]
    reordered_indices = absorbing_matrix_data["reordered_indices"]
    reordered_states = absorbing_matrix_data["reordered_states"]
    n_transient = absorbing_matrix_data["n_transient"]

    st.markdown(
        r"""
        After reordering the states, transient states are placed first and absorbing states
        are placed last. The transition matrix has the block form

        $$
        P =
        \begin{pmatrix}
        Q & R \\
        0 & I
        \end{pmatrix}.
        $$

        The block \(Q\) contains transitions among transient states.  
        The block \(R\) contains transitions from transient states to absorbing states.  
        The block \(I\) represents the absorbing victim and hazard states.
        """
    )

    labels = [
        f"s{original_idx}\n{state}"
        for original_idx, state in zip(reordered_indices, reordered_states)
    ]

    P_reordered_df = pd.DataFrame(
        P_reordered,
        index=labels,
        columns=labels,
    )

    st.plotly_chart(
        draw_absorbing_matrix_heatmap(
            P_reordered=P_reordered,
            labels=labels,
            n_transient=n_transient,
        ),
        use_container_width=True,
        key="absorbing_reordered_matrix_heatmap",
    )

        
   
    
    if len(transient_indices) > 0:
        # Extract Q and R matrices
        Q_abs = P[np.ix_(transient_indices, transient_indices)]
        R_abs = P[np.ix_(transient_indices, absorbing_indices)]
        
        
        q_columns = [f"s{idx}" for idx in transient_indices]
        q_index = [f"s{idx}" for idx in transient_indices]
        
        
        # Create unique labels for absorbing states
        r_columns = []
        for idx in absorbing_indices:
            state = states[idx]
            if is_victim(grid, state):
                r_columns.append(f"s{idx}_Success")
            elif is_hazard(grid, state):
                r_columns.append(f"s{idx}_Failure")
            else:
                r_columns.append(f"s{idx}")
        
        # Make column names unique if duplicates exist
        seen = {}
        unique_r_columns = []
        for col in r_columns:
            if col in seen:
                seen[col] += 1
                unique_r_columns.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_r_columns.append(col)
        
        r_index = [f"s{idx}" for idx in transient_indices]
        
       
        # Compute fundamental matrix N = (I - Q)^(-1)
        I = np.eye(len(Q_abs))
        
        try:
            N = np.linalg.inv(I - Q_abs)
            
            st.subheader("Fundamental Matrix N = (I - Q)⁻¹")

            st.markdown(
                """
                After separating transient and absorbing states, the transition matrix can be written
                using the transient-to-transient block $Q$.

                The **fundamental matrix** is defined as:

                $$
                N = (I - Q)^{-1}
                $$

                The entry $N_{ij}$ represents the **expected number of times** the process visits
                transient state $s_j$ before absorption, given that it started from transient state $s_i$.

                In other words, each row answers the question:

                **If the robot starts from this transient state, how many times do we expect it to visit
                each transient state before reaching either the victim or a hazard?**

                Large values indicate states that are expected to be visited often before absorption.
                Smaller values indicate states that are rarely visited before the mission ends.
                """
            )
            
            st.plotly_chart(
                draw_fundamental_matrix_heatmap(
                    N=N,
                    transient_indices=transient_indices,
                    states=states,
                ),
                use_container_width=True,
                key="fundamental_matrix_heatmap",
            )


            # Expected steps until absorption
            expected_steps = N.sum(axis=1)

            expected_df = pd.DataFrame({
                "State": [f"s{i}" for i in transient_indices],
                "Position": [str(states[i]) for i in transient_indices],
                "Expected Steps Until Absorption": np.round(expected_steps, 3),
            })

            st.markdown(
                """
                ### Expected Steps Until Absorption

                The fundamental matrix $N$ tells us how often each transient state is expected
                to be visited before absorption.

                To obtain the expected number of steps before absorption, **we sum each row of $N$**:

                $$
                \\tau = N \\cdot \\mathbf{1}
                $$

                where $\\mathbf{1}$ is a column vector of ones.

                Therefore, each value $\\tau_i$ represents the expected number of steps before the
                robot is absorbed, starting from transient state $s_i$.

                Larger values mean that the robot is expected to move for longer before the mission
                ends. Smaller values mean that absorption is expected to happen sooner.
                """
            )

            col_expected_grid, col_expected_table = st.columns([1, 1])

            col_expected_grid, col_expected_table = st.columns([1, 1])

            with col_expected_grid:
                st.markdown("#### Expected steps on the grid")

                st.plotly_chart(
                    draw_expected_steps_grid_heatmap(
                        grid=grid,
                        states=states,
                        transient_indices=transient_indices,
                        expected_steps=expected_steps,
                        start_pos=start_pos,
                    ),
                    use_container_width=True,
                    key="expected_steps_grid_heatmap",
                )

            with col_expected_table:
                if start_pos in states:
                    start_idx = states.index(start_pos)

                    if start_idx in transient_indices:
                        start_pos_in_transient = transient_indices.index(start_idx)
                        expected_from_start = expected_steps[start_pos_in_transient]

                        st.markdown("#### Start-state result")

                        html_card = f"""
            <div style="background-color:#f8fafc; border:2px solid #2563eb; border-radius:12px; padding:22px; margin-top:20px; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                <div style="font-size:15px; color:#475569; margin-bottom:8px;">
                    Starting from
                </div>
                <div style="font-size:24px; font-weight:700; color:#1e293b; margin-bottom:16px;">
                    state s{start_idx} &nbsp; | &nbsp; position {start_pos}
                </div>
                <div style="font-size:15px; color:#475569; margin-bottom:8px;">
                    Expected steps until absorption
                </div>
                <div style="font-size:38px; font-weight:800; color:#2563eb; margin-bottom:10px;">
                    {expected_from_start:.2f} steps
                </div>
                <div style="font-size:14px; color:#475569; line-height:1.5;">
                    This is the expected number of movement steps before the robot reaches either the victim state or a hazard state.
                </div>
            </div>
            """

                        st.markdown(html_card, unsafe_allow_html=True)
                        
            # Absorption probabilities B = N @ R
            B = N @ R_abs
            
            # Prepare unique labels for absorbing states
            absorbing_labels = []
            absorbing_state_info = []
            
            for i, idx in enumerate(absorbing_indices):
                state = states[idx]
                if is_victim(grid, state):
                    label = f"Success_s{idx}"
                    absorbing_state_info.append(("success", idx, label))
                elif is_hazard(grid, state):
                    label = f"Failure_s{idx}"
                    absorbing_state_info.append(("failure", idx, label))
                else:
                    label = f"s{idx}"
                    absorbing_state_info.append(("other", idx, label))
                
                absorbing_labels.append(label)
            
            # Make absorbing labels unique
            seen_labels = {}
            unique_absorbing_labels = []
            for label in absorbing_labels:
                if label in seen_labels:
                    seen_labels[label] += 1
                    unique_absorbing_labels.append(f"{label}_{seen_labels[label]}")
                else:
                    seen_labels[label] = 0
                    unique_absorbing_labels.append(label)
            
            absorption_df = pd.DataFrame(
                np.round(B, 4),
                columns=unique_absorbing_labels
            )
            
            absorption_df.insert(
                0,
                "Transient State",
                [f"s{i} - {str(states[i])}" for i in transient_indices]
            )
            
            st.markdown("#### Absorption Probabilities $B = N R$")

            st.markdown(
                """
                The matrix $B$ gives the probabilities of ending in each absorbing state.

                It is computed as:

                $$
                B = N R
                $$

                where:

                - $N$ is the fundamental matrix;
                - $R$ contains the transition probabilities from transient states to absorbing states.

                Each row of $B$ corresponds to a transient starting state.
                Each column corresponds to an absorbing state.

                Therefore, the entry $B_{ij}$ represents the probability that the robot is eventually
                absorbed in absorbing state $j$, given that it started from transient state $i$.

                In this rescue problem, absorbing states correspond to:

                - **success**, if the robot reaches the victim;
                - **failure**, if the robot enters a hazard.

                For each transient state, the absorption probabilities across all absorbing states sum to 1.
                """
            )

            col1,col2=st.columns(2)

            with col1:
            
                st.plotly_chart(
                    draw_absorption_probabilities_heatmap(
                        B=B,
                        transient_indices=transient_indices,
                        absorbing_indices=absorbing_indices,
                        states=states,
                        grid=grid,
                    ),
                    use_container_width=True,
                    key="absorption_probabilities_heatmap",
                )

            
            if start_pos in states:
                start_idx = states.index(start_pos)

                if start_idx in transient_indices:
                    start_pos_in_transient = transient_indices.index(start_idx)

                    start_absorption_probs = B[start_pos_in_transient, :]

                    success_prob_start = 0.0
                    failure_prob_start = 0.0

                    for local_abs_idx, global_abs_idx in enumerate(absorbing_indices):
                        abs_state = states[global_abs_idx]

                        if is_victim(grid, abs_state):
                            success_prob_start += start_absorption_probs[local_abs_idx]

                        elif is_hazard(grid, abs_state):
                            failure_prob_start += start_absorption_probs[local_abs_idx]

                    with col2:

                        st.markdown("#### Start-state absorption result")

                        html_card = f"""
                <div style="background-color:#f8fafc; border:2px solid #2563eb; border-radius:12px; padding:22px; margin-top:20px; box-shadow:0 2px 8px rgba(0,0,0,0.06);">
                    <div style="font-size:15px; color:#475569; margin-bottom:8px;">
                        Starting from
                    </div>
                    <div style="font-size:24px; font-weight:700; color:#1e293b; margin-bottom:16px;">
                        state s{start_idx} &nbsp; | &nbsp; position {start_pos}
                    </div>
                    <div style="font-size:15px; color:#475569; margin-bottom:8px;">
                        Probability of eventual success
                    </div>
                    <div style="font-size:38px; font-weight:800; color:#16a34a; margin-bottom:12px;">
                        {success_prob_start:.4f}
                    </div>
                    <div style="font-size:15px; color:#475569; margin-bottom:8px;">
                        Probability of eventual failure
                    </div>
                    <div style="font-size:32px; font-weight:800; color:#dc2626; margin-bottom:12px;">
                        {failure_prob_start:.4f}
                    </div>
                    <div style="font-size:14px; color:#475569; line-height:1.5;">
                        These probabilities describe the final mission outcome when the robot starts from the initial state.
                        They sum to approximately 1.
                    </div>
                </div>
                """

                        st.markdown(html_card, unsafe_allow_html=True)
            
            # Additional statistics
            st.subheader("Summary Statistics")
            
            total_expected_steps = expected_steps.sum()
            avg_expected_steps = expected_steps.mean()
            max_expected_steps = expected_steps.max()
            max_state_idx = transient_indices[np.argmax(expected_steps)]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Expected Steps", f"{total_expected_steps:.1f}")
            with col2:
                st.metric("Average Expected Steps", f"{avg_expected_steps:.2f}")
            with col3:
                st.metric("Max Expected Steps", f"{max_expected_steps:.2f}")
            with col4:
                st.metric("Most Time-Consuming State", f"s{max_state_idx}")
            
            condition_number = np.linalg.cond(I - Q_abs)
            st.caption(f"Condition number of (I - Q): {condition_number:.2e}")
            if condition_number > 1e10:
                st.warning("Matrix is nearly singular - check if all transient states are truly transient")
            else:
                st.success("Matrix is well-conditioned")
                
        except np.linalg.LinAlgError as e:
            st.error(f"Matrix inversion failed: {e}")
            st.warning("This usually means (I - Q) is singular. Check if all transient states are correctly identified.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("No transient states found. The robot starts in an absorbing state.")
   


# ============================================================
# Tab 2: Continuous-time Markov chain
# ============================================================

with tab_continuous:
    st.header("2. Continuous-Time Markov Chain Approach")

   
    st.markdown(
    """
    In the continuous-time version, the robot does not move at fixed discrete steps.
    Instead, when the robot is in a state, it waits for a random amount of time and
    then moves to another state. 

    The continuous-time model is described by a generator matrix $Q$. Unlike the
    discrete transition matrix $P$, whose entries are probabilities, the entries of
    $Q$ are *transition rates*.

    **For two different states $s_i$ and $s_j$, the value $q_{ij}$ represents the
    instantaneous rate of moving from state $s_i$ to state $s_j$**. 
    
    It is computed from the discrete transition probability $P_{ij}$ by multiplying it by the
    transition rate parameter $\lambda$:

    $$
    q_{ij}=\\lambda P_{ij}, \\quad i \\neq j.
    $$

    The parameter $\lambda$ controls how fast transitions occur. A larger value of
    $\lambda$ means that the robot moves more quickly between states, while a smaller
    value means that the robot waits longer before moving.

    The diagonal entries of $Q$ are chosen so that each row sums to zero:

    $$
    q_{ii}=-\\sum_{j\\neq i} q_{ij}.
    $$

    **Therefore, $q_{ii}$ is negative and represents the total rate of leaving state
    $s_i$**.

    Absorbing states, such as the victim state and hazard states, have zero rows in
    $Q$. This means that once the robot reaches one of these states, it remains
    there forever.
    """
)

    P, states = build_transition_matrix(
        grid=grid,
        policy_type=policy_type,
        success_prob=success_prob,
        victim_pos=victim_pos,
    )

    Q = build_generator_matrix_from_transition_matrix(
        P=P,
        states=states,
        grid=grid,
        transition_rate=transition_rate,
    )

    # ============================================================
    # Environment and discrete transition matrix before Q
    # ============================================================

    st.markdown("#### Rescue Environment")

    render_frozenlake_state(
        grid=grid,
        robot_pos=start_pos,
        caption=(
            f"Initial configuration | Start: {start_pos} | "
            f"Victim: {victim_pos}"
        ),
        image_width=300,
    )


    col1, col2 = st.columns(2)

    
    with col1:
        st.markdown("#### Discrete-Time Transition Matrix $P$")

        st.markdown(
            """
            The entry $P_{ij}$ is the probability of moving
            from $s_i$ to $s_j$ in one discrete step.
            """
        )

        start_index = states.index(start_pos)

        st.plotly_chart(
            draw_transition_matrix_heatmap(
                P,
                states,
                start_state_idx=start_index,
            ),
            use_container_width=True,
            key="continuous_tab_discrete_transition_matrix_P",
        )

    with col2:

        st.markdown("#### Generator Matrix $Q$")

        st.markdown(
            """
            The generator matrix $Q$ is  built from  matrix $P$ by converting probabilities into transition rates.
            """
        )


        st.plotly_chart(
            draw_generator_matrix_heatmap(Q, states),
            use_container_width=True,
        )
        st.write("Each row of a valid generator matrix should sum to 0.")

    st.markdown("---")

    
    st.subheader("Continuous-Time Transition Matrix $P(t)=e^{Qt}$")



    st.markdown(
        """
        The generator matrix $Q$ does not directly give transition probabilities over a
        finite time interval. It gives only the instantaneous transition rates.

        To obtain the probability of being in each state after a continuous time $t$,
        we compute the continuous-time transition matrix:

        $$
        P(t) = e^{Qt}.
        $$

        This matrix is obtained using the matrix exponential:

        $$
        e^{Qt}
        =
        I + Qt + \\frac{(Qt)^2}{2!}
        + \\frac{(Qt)^3}{3!}
        + \\cdots
        $$

        Each entry $P_{ij}(t)$ represents the probability that the robot is in state
        $s_j$ at time $t$, given that it started in state $s_i$ at time $0$:

        $$
        P_{ij}(t)
        =
        P(X(t)=s_j \\mid X(0)=s_i).
        $$

        Therefore, each row of $P(t)$ is a probability distribution and should sum to 1.

        The key interpretation is:

        **$Q$ tells how fast transitions happen.**

        **$P(t)$ tells where the robot may be after time $t$.**
        """
    )



   
    time_t = st.slider(
        "Time horizon t",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key="continuous_time_slider_t",
    )

    P_t = transition_matrix_over_time(Q, time_t)

    start_index = states.index(start_pos)
    probability_from_start = P_t[start_index, :]

    most_likely_idx = int(np.argmax(probability_from_start))
    most_likely_state = states[most_likely_idx]
    most_likely_prob = probability_from_start[most_likely_idx]

    col_pt_heatmap, col_pt_distribution = st.columns([2, 1])

    with col_pt_heatmap:
        st.markdown("#### Transition matrix $P(t)$")

        st.plotly_chart(
            draw_continuous_transition_matrix_heatmap(
                P_t=P_t,
                states=states,
                start_state_idx=start_index,
                most_likely_idx=most_likely_idx,
            ),
            use_container_width=True,
            key=f"continuous_transition_matrix_heatmap_t_{time_t}",
        )

    with col_pt_distribution:
        st.markdown("#### Distribution from initial state")

        st.plotly_chart(
            draw_state_probability_grid(
                grid=grid,
                states=states,
                distribution=probability_from_start,
                current_pos=most_likely_state,
                start_pos=start_pos,
            ),
            use_container_width=True,
            key=f"continuous_distribution_grid_t_{time_t}",
        )

        st.metric(
            label=f"Most likely state at time t = {time_t:.1f}",
            value=f"s{most_likely_idx}",
            delta=f"Probability {most_likely_prob:.4f}",
        )
       

    

    st.subheader("Animated Continuous-Time Mission Evolution")

    st.markdown(
        """
        The animation below synchronizes the main continuous-time quantities.

        For each time value $t$, the app computes:

        $$
        P(t) = e^{Qt}
        $$

        and then updates the state distribution:

        $$
        \\pi(t) = \\pi_0 P(t)
        $$

        The top-left plot shows the probability distribution on the rescue grid.
        The top-right plot shows the same distribution as a bar plot.
        The bottom plot shows the success, failure, and survival probabilities over time.
        """
    )

    pi0 = np.zeros(len(states))
    pi0[start_index] = 1.0

    victim_index = states.index(victim_pos)

    hazard_indices = [
        i for i, s in enumerate(states)
        if is_hazard(grid, s)
    ]

    st.plotly_chart(
        draw_continuous_process_animation(
            Q=Q,
            states=states,
            grid=grid,
            start_index=start_index,
            victim_index=victim_index,
            hazard_indices=hazard_indices,
            transition_matrix_over_time=transition_matrix_over_time,
            t_min=0.0,
            t_max=10.0,
            dt=0.1,
        ),
        use_container_width=True,
        key="synchronized_continuous_time_animation",
    )