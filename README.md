# 🧠 AI Brain Explorer (Transformer Visualizer)

> **An interactive Streamlit application that demystifies how a Transformer processes sequences, computes attention, and generates contextual representations.**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🌟 Overview

**AI Brain Explorer** is an educational and intuitive tool designed to visually break down the internal mechanics of Transformer models. From tokenization to layer-wise representations, this app allows you to explore exactly how Artificial Intelligence reads, understands, and forms connections between words.

## ✨ Features

The application is logically divided into 7 core visual components, each representing a crucial step in the Transformer architecture:

1. **📝 Word Splitting (Tokenization & Embeddings)**: See how sentences are split into individual tokens and converted into numerical "Meaning Vectors".
2. **📐 Word Order (Positional Encoding)**: Understand how the model distinguishes word sequences using unique sine/cosine mathematical wave patterns.
3. **🔗 Word Connections (Self-Attention)**: Dive into Query, Key, and Value vectors. Visualize raw attention scores and explore the interactive heatmap of softmax attention weights.
4. **🧠 Finding Patterns (Multi-Head Attention)**: Compare multiple attention heads in parallel and see how different heads focus on different syntactic or semantic relationships.
5. **⚡ Processing Meaning (Feed-Forward Network)**: Explore the position-wise FFN and see how the ReLU activation function introduces non-linearity.
6. **♻️ Brain Polish (Residuals & Layer Norm)**: Visualize network stabilization. See how skip connections prevent vanishing gradients, and how LayerNorm centers vector distributions.
7. **📊 Deep Understanding (Layer Stacking)**: Track token vectors across multiple consecutive layers. Observe how vector representations become increasingly contextual via Cosine Similarity and L2 Norm drop-offs.

## 🚀 Getting Started

### Prerequisites

You need Python installed on your machine. Install the required dependencies using `pip`:

```bash
pip install streamlit numpy pandas plotly
```

### Running the App

1. Navigate to the project directory:
```bash
cd Streamlint-Demo
```

2. Run the Streamlit server:
```bash
streamlit run app.py
```

3. The application will launch automatically in your default internet browser at `http://localhost:8501`.

## ⚙️ Interactive Sidebar

- **✏️ Input Sentence**: Try out different sentences to see how the attention networks and embeddings react to changing vocabulary.
- **⚙️ Hyperparameters**: Adjust the Number of Layers, Attention Heads, and Embedding Dimension on the fly to see how it affects the matrices.
- **🔘 Display Toggles**: Toggle the visibility of complex graphs like Positional Encodings and Attention Maps.

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)**: For the highly interactive, responsive web UI and custom CSS styling.
- **[Plotly Express & Graph Objects](https://plotly.com/python/)**: For generating beautiful, scalable dark-themed heatmaps, bar charts, and histograms.
- **[NumPy & Pandas](https://numpy.org/)**: For constructing the underlying matrix multiplications, linear algebra operations, and data logic safely from scratch.

---
*Built with ❤️ to make AI approachable and visually understandable.*
