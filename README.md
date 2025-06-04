# 🌲 TRW-GBP Pose Graph Optimizer

An interactive demo to visualize **Tree-Reweighted Gaussian Belief Propagation (TRW-GBP)** on 2D pose graphs.
It accelerates and converges to standard GBP by averaging spanning tree GBP beliefs.

---

## ⚙️ Installation

```bash
git clone https://github.com/YuzhouCheng66/trw_gbp.git
cd trw_gbp
pip install -r requirements.txt
streamlit run demo.py
```

---

## 🧭 UI Overview

### Construction Settings
- **Random Seed**: Set a seed to ensure reproducible graphs.
- **Number of Nodes**: Controls how many nodes are generated in the graph.
- **Odometry Noise Std**: Adjust the Gaussian noise in edge measurements.
- **Reconstruct**: Rebuild the pose graph using current parameters.

### Optimization Settings
- **Number of Spanning Trees**: Choose how many random spanning trees to generate for TRW-GBP.
- **Run Optimization**: Starts tree sampling, GBP solving, and TRW averaging.

### Display Options
- **Show Ground Truth**: Display true 2D positions.
- **Show Full Optimization**: Display solution from full information matrix.
- **Show Tree**: Toggle a specific tree's optimization result.
  - **Tree Index**: Choose which sampled tree to display.

### TRW-GBP Tree Count
- Adjust the number of trees used in the TRW-GBP average.
- Watch as green TRW-GBP estimates converge to blue full optimization.

---

## 📊 Visualization

- **Top Graph**: Shows GT (●), Full (❌), Anchor (🔴), TRW estimate (✳️), and edges (gray lines).
- **Bottom Plot**: Convergence plot of TRW-GBP vs. Full L2 error as tree count increases.

---

## 💡 Notes

- TRW-GBP converges to full-batch as number of trees → ∞.
- Tree samples and noise injection are repeatable if seed is fixed.
- See `requirements.txt` for full list (Streamlit, NumPy, Plotly, SciPy).
---


Have fun! 