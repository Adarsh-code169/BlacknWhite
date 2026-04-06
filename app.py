"""
Transformer Visualizer — An interactive Streamlit application that lets users
explore how a Transformer processes sequences, computes attention, and
generates contextual representations.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# Page config & custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Brain Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* ── Google‑Font import ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0f1117;
    --bg-card: #1a1d2e;
    --accent: #7c3aed;
    --accent-light: #a78bfa;
    --accent-glow: rgba(124, 58, 237, .35);
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: rgba(148,163,184,.12);
    --success: #34d399;
    --warning: #fbbf24;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-light) !important;
}

/* ── Cards ── */
div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,.25);
    transition: box-shadow .3s ease;
}
div[data-testid="stExpander"]:hover {
    box-shadow: 0 0 28px var(--accent-glow);
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .95rem !important;
    letter-spacing: .3px;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
}

/* ── Info / takeaway boxes ── */
.takeaway-box {
    background: linear-gradient(135deg, rgba(124,58,237,.12), rgba(99,102,241,.08));
    border-left: 4px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0 20px 0;
    color: var(--text);
    font-size: .92rem;
    line-height: 1.6;
}

.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 4px;
    color: var(--accent-light);
}

/* Hide default streamlit header / footer */
header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────

def _set_seed(text: str, extra: int = 0):
    """Deterministic random state so visuals stay stable across reruns."""
    np.random.seed(hash(text + str(extra)) % (2**31))


def _takeaway(msg: str):
    st.markdown(f'<div class="takeaway-box">🔍 <b>What should you observe?</b><br>{msg}</div>', unsafe_allow_html=True)


def _section_intro(text: str):
    st.markdown(f"<p style='color:#94a3b8;font-size:.93rem;line-height:1.65'>{text}</p>", unsafe_allow_html=True)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: d_model // 2])   # handle odd d_model
    return pe


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🤖 AI Brain Explorer")
    st.markdown("---")
    input_text = st.text_area(
        "✏️ Input Sentence",
        value="The cat sat on the mat",
        height=90,
        help="Enter a sentence to tokenize and process through the Transformer pipeline.",
    )
    st.markdown("### ⚙️ Hyperparameters")
    num_layers = st.slider("Number of Layers", 1, 6, 2)
    num_heads = st.slider("Number of Attention Heads", 1, 8, 4)
    d_model = st.slider("Embedding Dimension", 8, 64, 32, step=4)
    st.markdown("---")
    st.markdown("### 🔘 Display Toggles")
    show_attention = st.toggle("Show Attention Maps", value=True)
    show_pe = st.toggle("Show Positional Encoding", value=True)
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & Plotly")

# ──────────────────────────────────────────────
# Derived data
# ──────────────────────────────────────────────

tokens = input_text.strip().split()
n_tokens = len(tokens)

_set_seed(input_text)
embeddings = np.random.randn(n_tokens, d_model).astype(np.float32) * 0.5
pe = positional_encoding(n_tokens, d_model).astype(np.float32)
embedded_input = embeddings + pe

# ──────────────────────────────────────────────
# Title
# ──────────────────────────────────────────────

st.markdown(
    """
    <h1 style='text-align:center; background: linear-gradient(90deg,#7c3aed,#6366f1,#818cf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.4rem;
    font-weight:800; margin-bottom:0'>AI Brain Explorer</h1>
    <p style='text-align:center;color:#94a3b8;margin-top:4px;font-size:1rem'>
    Discover exactly how Artificial Intelligence reads, understands, and forms connections between words.</p>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# Quick stats
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tokens", n_tokens)
c2.metric("Embed Dim", d_model)
c3.metric("Heads", num_heads)
c4.metric("Layers", num_layers)

st.markdown("---")

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📝 1. Word Splitting",
    "📐 2. Word Order",
    "🔗 3. Word Connections",
    "🧠 4. Finding Patterns",
    "⚡ 5. Processing Meaning",
    "♻️ 6. Brain Polish",
    "📊 7. Deep Understanding",
])

# ======== TAB 1 — TOKENIZATION ========
with tab1:
    st.markdown("<div class='section-header'>Step 1: Splitting and Converting Words to Numbers</div>", unsafe_allow_html=True)
    _section_intro(
        "AI cannot read text like humans do. First, sentences are split into individual parts called <b>Tokens</b>. "
        "Then, each token is converted into a list of numbers (a <b>Meaning Vector</b>). This is how the AI begins to 'read'."
    )

    col_tok, col_emb = st.columns([1, 2])

    with col_tok:
        st.markdown("#### Words to ID Numbers")
        tok_df = pd.DataFrame({"Word (Token)": tokens, "ID Number": list(range(n_tokens))})
        st.dataframe(tok_df, use_container_width=True, hide_index=True)

    with col_emb:
        st.markdown("#### Word Meaning Vectors (Color Map)")
        fig = px.imshow(
            embeddings,
            labels=dict(x="List of Numbers (Features)", y="Word", color="Meaning Value"),
            y=tokens,
            color_continuous_scale="Viridis",
            aspect="auto",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            height=max(250, n_tokens * 45),
        )
        st.plotly_chart(fig, use_container_width=True)

    _takeaway(
        "Each token is represented as a row vector. Similar words may have similar embedding "
        "patterns once a real model is trained — here we use random embeddings for illustration."
    )

# ======== TAB 2 — POSITIONAL ENCODING ========
with tab2:
    st.markdown("<div class='section-header'>Step 2: Adding Word Order (Position Patterns)</div>", unsafe_allow_html=True)
    _section_intro(
        "Because AI reads everything at the exact same time, it doesn't automatically know what order the words are in! "
        "To fix this, a unique <b>wavy math pattern</b> is added to each word based on its timeline position."
    )

    if show_pe:
        pe_col1, pe_col2 = st.columns(2)

        with pe_col1:
            st.markdown("#### Positional Encoding Pattern")
            fig_pe = px.imshow(
                pe,
                labels=dict(x="Dimension", y="Position", color="Value"),
                y=[f"pos {i} ({t})" for i, t in enumerate(tokens)],
                color_continuous_scale="RdBu_r",
                aspect="auto",
            )
            fig_pe.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                height=max(250, n_tokens * 45),
            )
            st.plotly_chart(fig_pe, use_container_width=True)

        with pe_col2:
            st.markdown("#### The Final Mixed Information (Meaning + Order)")
            fig_comb = px.imshow(
                embedded_input,
                labels=dict(x="List of Numbers", y="Word", color="Power Size"),
                y=tokens,
                color_continuous_scale="Magma",
                aspect="auto",
            )
            fig_comb.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                height=max(250, n_tokens * 45),
            )
            st.plotly_chart(fig_comb, use_container_width=True)

        # Sine / cosine wave plot for first few dims
        st.markdown("#### Sine & Cosine Waves (first 4 dimensions)")
        max_pos = max(n_tokens, 20)
        pe_long = positional_encoding(max_pos, d_model)
        wave_df = pd.DataFrame({
            "Position": list(range(max_pos)) * 4,
            "Value": np.concatenate([pe_long[:, i] for i in range(4)]),
            "Dimension": sum([[f"dim {i}"] * max_pos for i in range(4)], []),
        })
        fig_wave = px.line(wave_df, x="Position", y="Value", color="Dimension",
                           template="plotly_dark")
        fig_wave.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=0, r=0, t=20, b=0), height=300)
        st.plotly_chart(fig_wave, use_container_width=True)
    else:
        st.info("Toggle **Show Positional Encoding** in the sidebar to visualize.")

    _takeaway(
        "Even-numbered dimensions use sine and odd-numbered dimensions use cosine. Each position "
        "gets a unique pattern, enabling the Transformer to distinguish word order."
    )

# ======== TAB 3 — SELF-ATTENTION ========
with tab3:
    st.markdown("<div class='section-header'>Self-Attention Mechanism</div>", unsafe_allow_html=True)
    _section_intro(
        "Self-attention lets every token look at <b>every other token</b> in the sequence. "
        "Each token creates a <b>Query</b>, <b>Key</b>, and <b>Value</b> vector. The attention "
        "score between two tokens is the dot product of their Query and Key, scaled by √d."
    )

    if show_attention:
        _set_seed(input_text, extra=1)
        W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.3
        W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.3
        W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.3

        Q = embedded_input @ W_q
        K = embedded_input @ W_k
        V = embedded_input @ W_v

        scores = Q @ K.T / np.sqrt(d_model)
        attn_weights = softmax(scores)
        attn_output = attn_weights @ V

        att_col1, att_col2 = st.columns(2)

        with att_col1:
            st.markdown("#### Raw Attention Scores (Q·Kᵀ / √d)")
            fig_raw = px.imshow(
                scores,
                labels=dict(x="Key Token", y="Query Token", color="Score"),
                x=tokens, y=tokens,
                color_continuous_scale="Inferno",
                aspect="auto",
            )
            fig_raw.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                height=max(300, n_tokens * 50),
            )
            st.plotly_chart(fig_raw, use_container_width=True)

        with att_col2:
            st.markdown("#### Attention Weights (after Softmax)")
            fig_attn = px.imshow(
                attn_weights,
                labels=dict(x="Key Token", y="Query Token", color="Weight"),
                x=tokens, y=tokens,
                color_continuous_scale="Purples",
                aspect="auto",
                text_auto=".2f",
            )
            fig_attn.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=30, b=0),
                height=max(300, n_tokens * 50),
            )
            st.plotly_chart(fig_attn, use_container_width=True)

        # Interactive token picker
        st.markdown("#### 🎯 Explore Word Connections")
        selected_token = st.selectbox("Select a word to see what it pays attention to:", tokens, key="sa_pick")
        idx = tokens.index(selected_token)
        weight_df = pd.DataFrame({"Word": tokens, "Attention Strength": attn_weights[idx]})
        fig_bar = px.bar(weight_df, x="Word", y="Attention Strength", color="Attention Strength",
                         color_continuous_scale="Purples", template="plotly_dark")
        fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=0, t=20, b=0), height=280,
                              showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("Toggle **Show Attention Maps** in the sidebar to visualize.")

    _takeaway(
        "After softmax, each row sums to 1 — it's a probability distribution. Tokens that are "
        "semantically or syntactically related tend to attend to each other more strongly."
    )

# ======== TAB 4 — MULTI-HEAD ATTENTION ========
with tab4:
    st.markdown("<div class='section-header'>Multi-Head Attention</div>", unsafe_allow_html=True)
    _section_intro(
        "Instead of one set of Q, K, V projections, the model uses <b>multiple heads</b> in "
        "parallel. Each head can learn <b>different</b> relationships — e.g., one head may "
        "focus on syntactic dependencies while another captures semantic similarity."
    )

    if show_attention:
        d_head = max(d_model // num_heads, 2)
        head_attns = []

        for h in range(num_heads):
            _set_seed(input_text, extra=100 + h)
            Wq_h = np.random.randn(d_model, d_head).astype(np.float32) * 0.3
            Wk_h = np.random.randn(d_model, d_head).astype(np.float32) * 0.3
            Qh = embedded_input @ Wq_h
            Kh = embedded_input @ Wk_h
            sc = Qh @ Kh.T / np.sqrt(d_head)
            head_attns.append(softmax(sc))

        # Display heads in a grid
        cols_per_row = min(num_heads, 4)
        rows_needed = (num_heads + cols_per_row - 1) // cols_per_row

        for r in range(rows_needed):
            cols = st.columns(cols_per_row)
            for c_idx, col in enumerate(cols):
                h_idx = r * cols_per_row + c_idx
                if h_idx >= num_heads:
                    break
                with col:
                    st.markdown(f"**Head {h_idx + 1}**")
                    fig_h = px.imshow(
                        head_attns[h_idx],
                        x=tokens, y=tokens,
                        color_continuous_scale="Tealgrn",
                        aspect="auto",
                    )
                    fig_h.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=max(220, n_tokens * 38),
                        xaxis_title="", yaxis_title="",
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

        # Head comparison
        st.markdown("#### 📊 Pattern Comparer — Which word does each brain part look at?")
        comp_token = st.selectbox("Select a target word:", tokens, key="mha_pick")
        comp_idx = tokens.index(comp_token)
        comp_data = []
        for h in range(num_heads):
            for t_i, t in enumerate(tokens):
                comp_data.append({"Pattern Finder": f"Finder {h+1}", "Word": t, "Attention Strength": float(head_attns[h][comp_idx, t_i])})
        comp_df = pd.DataFrame(comp_data)
        fig_comp = px.bar(comp_df, x="Word", y="Attention Strength", color="Pattern Finder", barmode="group",
                          template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=0, r=0, t=20, b=0), height=320)
        st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.info("Toggle **Show Attention Maps** in the sidebar to visualize.")

    _takeaway(
        "Each head learns a different attention pattern. Some heads may focus on adjacent words, "
        "others on long-range relationships. This diversity gives Transformers their power."
    )

# ======== TAB 5 — FEEDFORWARD NETWORK ========
with tab5:
    st.markdown("<div class='section-header'>Feed-Forward Network (FFN)</div>", unsafe_allow_html=True)
    _section_intro(
        "After attention, each token's representation passes through a <b>position-wise "
        "feed-forward network</b> — two linear transformations with a ReLU activation in between. "
        "This adds non-linearity and further reshapes each token's representation independently."
    )

    _set_seed(input_text, extra=200)
    d_ff = d_model * 4
    W1 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.2
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.2
    b2 = np.zeros(d_model)

    ffn_hidden = relu(embedded_input @ W1 + b1)
    ffn_output = ffn_hidden @ W2 + b2

    ff_col1, ff_col2 = st.columns(2)

    with ff_col1:
        st.markdown("#### Before FFN (Input)")
        fig_bf = px.imshow(embedded_input, y=tokens, color_continuous_scale="Viridis", aspect="auto",
                           labels=dict(x="Dim", y="Token", color="Value"))
        fig_bf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0),
                             height=max(250, n_tokens * 45))
        st.plotly_chart(fig_bf, use_container_width=True)

    with ff_col2:
        st.markdown("#### After FFN (Output)")
        fig_af = px.imshow(ffn_output, y=tokens, color_continuous_scale="Cividis", aspect="auto",
                           labels=dict(x="Dim", y="Token", color="Value"))
        fig_af.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0),
                             height=max(250, n_tokens * 45))
        st.plotly_chart(fig_af, use_container_width=True)

    # ReLU activation histogram
    st.markdown("#### ReLU Activation Distribution (hidden layer)")
    relu_vals = ffn_hidden.flatten()
    fig_relu = px.histogram(x=relu_vals, nbins=60, template="plotly_dark",
                            color_discrete_sequence=["#7c3aed"],
                            labels={"x": "Activation Value", "y": "Count"})
    fig_relu.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=20, b=0), height=260)
    st.plotly_chart(fig_relu, use_container_width=True)

    _takeaway(
        "The FFN expands dimensionality (d_model → 4×d_model), applies ReLU to introduce "
        "non-linearity, then projects back. Notice the many zero activations — that's ReLU at work!"
    )

# ======== TAB 6 — RESIDUAL & NORMALIZATION ========
with tab6:
    st.markdown("<div class='section-header'>Residual Connections & Layer Normalization</div>", unsafe_allow_html=True)
    _section_intro(
        "Residual (skip) connections add the <b>input</b> back to the <b>output</b> of each "
        "sub-layer, preventing vanishing gradients. <b>Layer Norm</b> then normalises each "
        "token's vector to have zero mean and unit variance, stabilising training."
    )

    residual = embedded_input + ffn_output
    normed = layer_norm(residual)

    rn_col1, rn_col2, rn_col3 = st.columns(3)

    with rn_col1:
        st.markdown("#### Input (x)")
        fig_x = px.imshow(embedded_input, y=tokens, color_continuous_scale="Viridis", aspect="auto")
        fig_x.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0),
                            height=max(220, n_tokens * 40))
        st.plotly_chart(fig_x, use_container_width=True)

    with rn_col2:
        st.markdown("#### Residual (x + FFN(x))")
        fig_res = px.imshow(residual, y=tokens, color_continuous_scale="Inferno", aspect="auto")
        fig_res.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0),
                              height=max(220, n_tokens * 40))
        st.plotly_chart(fig_res, use_container_width=True)

    with rn_col3:
        st.markdown("#### LayerNorm(x + FFN(x))")
        fig_ln = px.imshow(normed, y=tokens, color_continuous_scale="RdBu_r", aspect="auto")
        fig_ln.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0),
                             height=max(220, n_tokens * 40))
        st.plotly_chart(fig_ln, use_container_width=True)

    # Value range comparison
    st.markdown("#### Value Distribution Across Steps")
    dist_data = pd.DataFrame({
        "Step": (["Input"] * embedded_input.size +
                 ["Residual"] * residual.size +
                 ["LayerNorm"] * normed.size),
        "Value": np.concatenate([embedded_input.flatten(), residual.flatten(), normed.flatten()]),
    })
    fig_dist = px.box(dist_data, x="Step", y="Value", color="Step", template="plotly_dark",
                      color_discrete_sequence=["#6366f1", "#f59e0b", "#34d399"])
    fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=20, b=0), height=300, showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)

    _takeaway(
        "After Layer Norm, values are centred around 0 with a tighter spread. This stabilisation "
        "is critical for deep networks — without it, activations can explode or vanish."
    )

# ======== TAB 7 — LAYER-WISE REPRESENTATION ========
with tab7:
    st.markdown("<div class='section-header'>Stacking Layers & Contextual Representations</div>", unsafe_allow_html=True)
    _section_intro(
        "A Transformer stacks <b>multiple identical layers</b>. With each layer, token "
        "representations become increasingly <b>contextual</b> — early layers capture local "
        "syntax while later layers encode higher-level semantics."
    )

    # Simulate stacked layers
    layer_outputs = [embedded_input.copy()]
    current = embedded_input.copy()

    for layer_i in range(num_layers):
        _set_seed(input_text, extra=300 + layer_i * 10)
        Wq_l = np.random.randn(d_model, d_model).astype(np.float32) * 0.3
        Wk_l = np.random.randn(d_model, d_model).astype(np.float32) * 0.3
        Wv_l = np.random.randn(d_model, d_model).astype(np.float32) * 0.3

        Q_l = current @ Wq_l
        K_l = current @ Wk_l
        V_l = current @ Wv_l
        attn_l = softmax(Q_l @ K_l.T / np.sqrt(d_model)) @ V_l

        # Residual + Norm after attention
        current = layer_norm(current + attn_l)

        # FFN
        _set_seed(input_text, extra=400 + layer_i * 10)
        W1_l = np.random.randn(d_model, d_model * 4).astype(np.float32) * 0.2
        W2_l = np.random.randn(d_model * 4, d_model).astype(np.float32) * 0.2
        ffn_l = relu(current @ W1_l) @ W2_l

        # Residual + Norm after FFN
        current = layer_norm(current + ffn_l)
        layer_outputs.append(current.copy())

    # Heatmaps per layer
    st.markdown("#### Token Representations Across Layers")
    layer_cols = st.columns(min(num_layers + 1, 4))
    for li, lc in enumerate(layer_cols):
        if li > num_layers:
            break
        with lc:
            label = "Input" if li == 0 else f"Layer {li}"
            st.markdown(f"**{label}**")
            fig_layer = px.imshow(layer_outputs[li], y=tokens, color_continuous_scale="Plasma", aspect="auto")
            fig_layer.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                                    height=max(200, n_tokens * 35), coloraxis_showscale=False,
                                    xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_layer, use_container_width=True)

    # If more than 4 columns needed, show remaining
    if num_layers + 1 > 4:
        extra_cols = st.columns(min(num_layers + 1 - 4, 4))
        for li2, lc2 in enumerate(extra_cols):
            real_idx = 4 + li2
            if real_idx > num_layers:
                break
            with lc2:
                st.markdown(f"**Layer {real_idx}**")
                fig_layer2 = px.imshow(layer_outputs[real_idx], y=tokens, color_continuous_scale="Plasma", aspect="auto")
                fig_layer2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                         plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
                                         height=max(200, n_tokens * 35), coloraxis_showscale=False,
                                         xaxis_title="", yaxis_title="")
                st.plotly_chart(fig_layer2, use_container_width=True)

    # Track a single token through layers
    st.markdown("---")
    st.markdown("#### 🔬 Track a Token Across Layers")
    tracked = st.selectbox("Select a token to track:", tokens, key="layer_track")
    t_idx = tokens.index(tracked)

    norms = [float(np.linalg.norm(layer_outputs[l][t_idx])) for l in range(num_layers + 1)]
    norm_df = pd.DataFrame({
        "Layer": ["Input"] + [f"Layer {i+1}" for i in range(num_layers)],
        "L2 Norm": norms,
    })
    fig_norm = px.line(norm_df, x="Layer", y="L2 Norm", markers=True, template="plotly_dark",
                       color_discrete_sequence=["#a78bfa"])
    fig_norm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=0, r=0, t=20, b=0), height=280)
    st.plotly_chart(fig_norm, use_container_width=True)

    # Cosine similarity between token at first and each layer
    from numpy.linalg import norm as np_norm
    cos_sims = []
    base = layer_outputs[0][t_idx]
    for l in range(num_layers + 1):
        v = layer_outputs[l][t_idx]
        cos = float(np.dot(base, v) / (np_norm(base) * np_norm(v) + 1e-9))
        cos_sims.append(cos)

    cos_df = pd.DataFrame({
        "Layer": ["Input"] + [f"Layer {i+1}" for i in range(num_layers)],
        "Cosine Similarity to Input": cos_sims,
    })
    fig_cos = px.bar(cos_df, x="Layer", y="Cosine Similarity to Input", template="plotly_dark",
                     color="Cosine Similarity to Input", color_continuous_scale="Tealgrn")
    fig_cos.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=20, b=0), height=280)
    st.plotly_chart(fig_cos, use_container_width=True)

    _takeaway(
        "As the token passes through more layers, its representation diverges from the original "
        "embedding (cosine similarity drops). This means the model is incorporating more context "
        "from surrounding tokens with every layer."
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#64748b;font-size:.82rem'>"
    "AI Brain Explorer · Educational Demo · "
    "Built with Streamlit & Plotly</p>",
    unsafe_allow_html=True,
)