# Graph Signal Processing and Spectrogram Visualization via MinLA

## Overview

This project investigates the relationship between the **Minimum Linear Arrangement (MinLA)** problem and the **visualization of graph spectrograms**. It shows that solving MinLA provides meaningful ways to reorder columns in a graph spectrogram, enhancing interpretability. Conversely, optimizing the spectrogram visualization also leads to good MinLA solutions.

---

## What is Graph Signal Processing?

**Graph Signal Processing (GSP)** extends classical signal processing to data defined on graphs. A signal in this context assigns a value to each node of a graph. 

## Graph Spectrograms

In classical signal processing, a **spectrogram** shows how the frequency content of a signal evolves over time. For graphs, an analogous spectrogram is built by:

1. Choosing a **window function** (typically Gaussian-like).
2. Placing the window at different nodes of the graph.
3. Multiplying the window with the signal to localize it.
4. Applying the GFT to each windowed signal.
5. Building a matrix (spectrogram), where each **column corresponds to one node**

**Problem:** Unlike time-series, graphs do not have a natural node order, making the spectrogram column ordering arbitrary and often visually uninformative.

---

## The Column Ordering Problem

To visualize the spectrogram effectively, we need to **reorder its columns** so that similar frequency profiles are placed next to each other. However, since graph nodes have no inherent order, this becomes a combinatorial challenge.

---

## The Minimum Linear Arrangement (MinLA) Problem

The **MinLA** problem aims to assign integer labels (positions) to the nodes of a graph to **minimize the sum of absolute differences** between connected node labels:

Minimize: ∑_{(i, j) ∈ E} |ϕ(i) - ϕ(j)|


This naturally encourages adjacent nodes in the graph to be close in the ordering. A MinLA solution can thus serve as a candidate permutation for spectrogram columns.

---

## Spectrogram Similarity Measure

To assess the quality of a column permutation, we define a **similarity measure**:

\[
SM(S, P) = \sum_{j=1}^{N-1} \sum_{i=1}^{N} \left(1 - \left| S(i, P(j)) - S(i, P(j+1)) \right|^a \right)
\]

- \( S \): Spectrogram matrix  
- \( P \): Permutation of columns (i.e., node order)  
- \( a \): Norm degree (we use \( a = 1 \))  

A high similarity measure implies that adjacent columns (nodes) in the permutation are spectrally similar, resulting in visually coherent bands in the spectrogram.

---

## Key Findings

- **MinLA permutations produce spectrograms with higher similarity indicators** than random orderings.
- **Optimizing the similarity indicator leads to lower MinLA costs**.
- The **two problems are strongly correlated**: improvements in one often translate to the other.

---
