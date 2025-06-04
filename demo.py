import streamlit as st
import numpy as np
import random
import plotly.graph_objects as go
from scipy.spatial import Delaunay

st.set_page_config(layout="wide")

# Custom styling for padding and compact layout
st.markdown("""
    <style>
    .block-container {
        padding-top: 0.5rem !important;
    }
    header, footer {
        visibility: hidden;
    }
    .stNumberInput>div>input {
        max-width: 80px;
    }
    </style>
""", unsafe_allow_html=True)

# === Initialize session state ===
if 'positions' not in st.session_state:
    st.session_state.positions = None
    st.session_state.edges = None
    st.session_state.x_opt = None
    st.session_state.x_trees = None
    st.session_state.static_traces = None

# === Sidebar: Construction Settings ===
st.sidebar.header("Construction Settings")
seed = st.sidebar.number_input("Random Seed", value=0, step=1)
num_nodes = st.sidebar.slider("Number of Nodes", 10, 200, 100, step=1)
odom_noise_std = st.sidebar.slider("Odometry Noise Std", 0.0, 2.0, 0.5, step=0.01)
reconstruct_button = st.sidebar.button("Reconstruct")

# === Sidebar: Optimization Settings ===
st.sidebar.header("Optimization Settings")
num_trees = st.sidebar.slider("Number of Spanning Trees", 1, 500, 100, step=1)
run_button = st.sidebar.button("Run Optimization")
show_gt = st.sidebar.checkbox("Show Ground Truth", value=True)
show_full = st.sidebar.checkbox("Show Full Optimization", value=True)

# Show specific tree option with tight layout
col1, col2 = st.sidebar.columns([1, 1])
show_tree = col1.checkbox("Show Tree", value=False)
tree_index = col2.number_input(" ", value=1, step=1, min_value=1, max_value=num_trees, label_visibility="collapsed")

# === Graph Reconstruction ===
if reconstruct_button or st.session_state.positions is None:
    np.random.seed(seed)
    random.seed(seed+1)
    N, dim = num_nodes, 2
    positions = np.random.rand(N, dim) * 100
    tri = Delaunay(positions)
    edges = set()
    for triangle in tri.simplices:
        for i in range(3):
            a, b = triangle[i], triangle[(i + 1) % 3]
            edges.add(tuple(sorted((a, b))))
    edges = list(edges)

    # Build global info matrix and vector
    Lambda = np.zeros((N * dim, N * dim))
    eta = np.zeros(N * dim)
    Lambda[0:2, 0:2] += np.eye(2) * 1e7
    eta[0:2] += np.eye(2) @ positions[0] * 1e7

    for (i, j) in edges:
        mu_ij = positions[j] - positions[i]
        noise = np.random.randn(2) * odom_noise_std
        mu_ij_noisy = mu_ij + noise
        Omega = np.eye(2)
        J_i, J_j = -np.eye(2), np.eye(2)
        idx_i, idx_j = slice(2*i, 2*i+2), slice(2*j, 2*j+2)
        Lambda[idx_i, idx_i] += J_i.T @ Omega @ J_i
        Lambda[idx_i, idx_j] += J_i.T @ Omega @ J_j
        Lambda[idx_j, idx_i] += J_j.T @ Omega @ J_i
        Lambda[idx_j, idx_j] += J_j.T @ Omega @ J_j
        eta[idx_i] += J_i.T @ Omega @ mu_ij_noisy
        eta[idx_j] += J_j.T @ Omega @ mu_ij_noisy

    x_opt = np.linalg.solve(Lambda, eta).reshape(N, 2)

    # Static plot traces
    static_traces = []
    for (i, j) in edges:
        static_traces.append(go.Scatter(
            x=[positions[i][0], positions[j][0]],
            y=[positions[i][1], positions[j][1]],
            mode='lines', line=dict(color='lightgray', width=1),
            hoverinfo='skip', showlegend=False))

    if show_gt:
        static_traces.append(go.Scatter(
            x=positions[:, 0], y=positions[:, 1],
            mode='markers', name='GT',
            marker=dict(color='black', size=5, symbol='circle')))

    if show_full:
        static_traces.append(go.Scatter(
            x=x_opt[:, 0], y=x_opt[:, 1],
            mode='markers', name='Full',
            marker=dict(color='blue', size=6, symbol='x')))

    static_traces.append(go.Scatter(
        x=[positions[0, 0]], y=[positions[0, 1]],
        mode='markers', name='Anchor',
        marker=dict(color='red', size=10)))

    st.session_state.positions = positions
    st.session_state.edges = edges
    st.session_state.x_opt = x_opt
    st.session_state.x_trees = None
    st.session_state.static_traces = static_traces

# === Run Optimization ===
if run_button and st.session_state.positions is not None:
    positions = st.session_state.positions
    edges = st.session_state.edges
    x_opt = st.session_state.x_opt
    N = len(positions)

    adj_list = {i: [] for i in range(N)}
    for (i, j) in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    def sample_tree(adj):
        visited, tree_edges, stack = set(), [], [random.randrange(N)]
        visited.add(stack[0])
        while len(visited) < N:
            curr = stack[-1]
            neighbors = adj[curr]
            random.shuffle(neighbors)
            for nxt in neighbors:
                if nxt not in visited:
                    visited.add(nxt)
                    tree_edges.append(tuple(sorted((curr, nxt))))
                    stack.append(nxt)
                    break
            else:
                stack.pop()
        return tree_edges

    edge_count = {e: 0 for e in edges}
    trees = [sample_tree(adj_list) for _ in range(num_trees)]
    for tree in trees:
        for e in tree:
            edge_count[e] += 1

    rho_ij = {e: edge_count[e]/num_trees for e in edges}
    mu_ij_dict, scaled_weights = {}, {}
    for (i, j) in edges:
        mu = positions[j] - positions[i] + np.random.randn(2) * odom_noise_std
        mu_ij_dict[(i, j)] = mu
        mu_ij_dict[(j, i)] = -mu
        w = 1.0 / (rho_ij[(i, j)] if rho_ij[(i, j)] > 0 else 1e-6)
        scaled_weights[(i, j)] = scaled_weights[(j, i)] = w

    x_trees = np.zeros((num_trees, N, 2))
    I2 = np.eye(2)
    for k, tree in enumerate(trees):
        L_k, eta_k = np.zeros((N*2, N*2)), np.zeros(N*2)
        L_k[0:2, 0:2] += I2 * 1e7
        eta_k[0:2] += I2 @ positions[0] * 1e7
        for (i, j) in tree:
            mu, w = mu_ij_dict[(i, j)], scaled_weights[(i, j)]
            idx_i, idx_j = slice(2*i, 2*i+2), slice(2*j, 2*j+2)
            L_k[idx_i, idx_i] += I2 * w
            L_k[idx_j, idx_j] += I2 * w
            L_k[idx_i, idx_j] += -I2 * w
            L_k[idx_j, idx_i] += -I2 * w
            eta_k[idx_i] += -I2 @ mu * w
            eta_k[idx_j] +=  I2 @ mu * w
        x_k = np.linalg.solve(L_k, eta_k).reshape(N, 2)
        x_trees[k] = x_k

    st.session_state.x_trees = x_trees

# === Plot ===
if st.session_state.x_opt is not None:
    x_opt = st.session_state.x_opt
    positions = st.session_state.positions
    x_trees = st.session_state.x_trees
    static_traces = st.session_state.static_traces
    num_trees = len(x_trees) if x_trees is not None else 0

    fig = go.Figure()
    fig.update_layout(
        font=dict(size=14), height=700,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(scaleanchor='y', showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(orientation='h', yanchor='top', y=0.98, xanchor='right', x=1, font=dict(size=14))
    )

    for trace in static_traces:
        fig.add_trace(trace)

    x_trw_k = None
    if num_trees > 0:
        trw_k = st.sidebar.slider("TRW-GBP Tree Count", 1, num_trees, num_trees, step=1)
        x_trw_k = np.mean(x_trees[:trw_k], axis=0)
        fig.add_trace(go.Scattergl(
            x=x_trw_k[:, 0], y=x_trw_k[:, 1],
            mode='markers', name=f'TRW-GBP ({trw_k})',
            marker=dict(color='green', size=6, symbol='x')))

        # Show selected tree if requested
        if show_tree and 1 <= tree_index <= num_trees:
            x_single = x_trees[tree_index - 1]
            fig.add_trace(go.Scattergl(
                x=x_single[:, 0], y=x_single[:, 1],
                mode='markers', name=f'Tree {tree_index}',
                marker=dict(color='orange', size=4, symbol='triangle-up')))

        # Plot TRW error curve
        trw_errors = [np.linalg.norm(np.mean(x_trees[:k+1], axis=0) - x_opt) for k in range(num_trees)]
        error_fig = go.Figure()
        error_fig.add_trace(go.Scattergl(
            x=list(range(1, num_trees+1)),
            y=trw_errors,
            mode='lines+markers',
            name='TRW-GBP Error',
            line=dict(color='pink', width=3)
        ))
        error_fig.update_layout(
            title='TRW-GBP Convergence (L2 Error)',
            xaxis_title='Number of Trees',
            yaxis_title='L2 Error',
            height=300, margin=dict(t=40, b=20, l=10, r=10)
        )
    
    l2_error = np.linalg.norm(x_trw_k - x_opt) if x_trw_k is not None else float('nan')
    max_error = np.max(np.linalg.norm(x_opt - positions, axis=1))
    mean_error = np.mean(np.linalg.norm(x_opt - positions, axis=1))

    st.markdown(
        f"**Max Error (GT vs Full):** {max_error:.2e} &nbsp;&nbsp;&nbsp; "
        f"**Mean Error (GT vs Full):** {mean_error:.2e} &nbsp;&nbsp;&nbsp; "
        f"**L2 Error (TRW-GBP vs Full):** {l2_error:.2f}"
    )
    st.plotly_chart(fig, use_container_width=True)
    if num_trees > 0:
        st.plotly_chart(error_fig, use_container_width=True)
