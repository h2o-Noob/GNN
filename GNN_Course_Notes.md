# Graph Neural Networks — Complete Course Notes

*Primary references: William L. Hamilton, "Graph Representation Learning" (Morgan & Claypool, 2020) and Yao Ma & Jiliang Tang, "Deep Learning on Graphs" (Cambridge University Press, 2021). Notation below mostly follows Hamilton, since it's the notation you'll see in most PyTorch Geometric (PyG) documentation.*

---

# Part 1 — Foundations

## Lesson 1 — What is Graph Representation Learning?

### 1.1 The core idea

A graph is $G = (V, E)$: a set of nodes $V$ and a set of edges $E \subseteq V \times V$. Most real data with *relationships* — accounts sending money to accounts, users following users, atoms bonded to atoms, web pages linking to web pages — is naturally a graph, not a table.

**Graph representation learning (GRL)** is the problem of learning a mapping

$$f: v \rightarrow \mathbf{z}_v \in \mathbb{R}^d$$

that encodes each node (or edge, or whole graph) into a low-dimensional vector, such that the geometry of the vector space (distances, directions, dot products) reflects the structural and semantic relationships in the graph. Hamilton frames essentially the entire book around this "encoder–decoder" view: an **encoder** maps a node to an embedding, and a **decoder** tries to reconstruct some graph-related information (e.g., "are these two nodes connected?") from the embedding. Training pushes the encoder to produce embeddings that make the decoder's job possible.

Historically this was done with **shallow embeddings** (one learned vector per node, no reuse of a function — e.g., DeepWalk, node2vec, matrix-factorization methods). Modern GRL is dominated by **Graph Neural Networks (GNNs)**, which learn a *parametric function* that combines a node's own features with its neighbors' features — this function is shared across all nodes, so it generalizes to nodes and even entire graphs never seen during training. This "encoder is a function of the graph structure + features" idea is precisely what Lesson 2 (message passing) formalizes.

### 1.2 Why tabular models fail to capture relationships

A tabular model (logistic regression, XGBoost, a plain MLP) treats each row (e.g., each account, each transaction) as an **i.i.d. feature vector**. Three structural facts get thrown away:

1. **Relational dependence is invisible.** If account A is a shell company that routes money to account B, a tabular model sees two unrelated rows unless someone manually engineers a feature like "in-degree of A" or "total amount received from flagged accounts." GNNs learn these relational features automatically and *adaptively*, rather than requiring a human to hand-craft every possible relational statistic.
2. **Multi-hop / higher-order patterns are hard to hand-engineer.** Fraud rings, money-laundering "smurfing" patterns, or cascades in a social network are properties of *paths and subgraphs*, not single rows. A tabular model can only see what's manually aggregated into a feature (e.g., "# of 2-hop neighbors that are flagged") — and every new pattern of interest needs a new hand-built feature. A GNN's receptive field (Lesson 2) naturally grows with the number of layers, so multi-hop information is absorbed automatically.
3. **The graph itself carries information independent of node features.** Two nodes can have identical feature vectors but completely different roles because of *where* they sit in the graph (e.g., a hub vs. a leaf). Structural/positional information (degree, centrality, motifs, community membership) is exactly what tabular rows cannot express, but what graph structure encodes directly.

The practical symptom in fraud/AML work: a tabular model treats "this account sent $9,900 to another account" identically whether that account is a completely isolated one-off transaction or the 15th hop in a layered laundering chain — the *context* is lost.

### 1.3 Graph features vs. learned embeddings

There are two broad ways to turn graph structure into something a model can consume:

| | Hand-crafted graph features | Learned embeddings (GRL / GNNs) |
|---|---|---|
| What it is | Manually computed statistics: degree, PageRank, clustering coefficient, betweenness centrality, k-core number, motif counts | A dense vector $\mathbf{z}_v \in \mathbb{R}^d$ produced by a trainable encoder |
| Who decides what matters | A human analyst decides which statistics to compute | The model learns, from data and the downstream loss, what structural patterns matter |
| Generalization | Fixed formulas, don't adapt to the task | Adapt to whatever task/loss you train against |
| Cost | Cheap, interpretable, easy to audit | More expensive, less directly interpretable, but far more expressive |
| Limitation | Cannot capture interactions the analyst didn't think of | Requires enough data and careful training to avoid overfitting/oversmoothing |

In practice these are **not mutually exclusive** — many production pipelines use hand-crafted graph features as *inputs* to the node feature matrix $X$, which the GNN then further transforms. E.g., you might feed "account age," "in-degree," "total volume last 30 days" as the initial node features $h_v^{(0)}$, and let the GNN learn embeddings *on top of* those. Hamilton's book frames the "shallow embedding" methods (DeepWalk/node2vec, matrix factorization) as a bridge between the two: they're learned, but — unlike GNNs — they don't use node features at all and don't generalize to unseen nodes (they're **transductive**, not **inductive** — a distinction that becomes very important later, in Lesson 5).

### 1.4 Node, edge, and graph representations

GRL produces representations at three levels of granularity, each suited to a different type of downstream prediction:

- **Node representation** $\mathbf{z}_v$: one vector per node. Used for *node-level* tasks — e.g., classify an account as fraudulent/legitimate, predict a user's community/interest.
- **Edge representation**: typically computed from a pair of node embeddings, e.g. $\mathbf{z}_{uv} = g(\mathbf{z}_u, \mathbf{z}_v)$ where $g$ might be concatenation, Hadamard product, or a dedicated edge-feature vector combined with the two endpoint embeddings. Used for *edge/link-level* tasks — e.g., link prediction ("will these accounts transact?"), or, as in AML (Lesson 16), transaction classification.
- **Graph representation** $\mathbf{z}_G$: one vector for an entire graph (or subgraph), typically obtained by **pooling/readout** over all node embeddings (sum, mean, max, or a learned attention-weighted pooling). Used for *graph-level* tasks — e.g., classify a molecule as toxic/non-toxic, or — very relevant to AML — classify a *subgraph/transaction ring* as a laundering pattern (Lesson 16, Option C in Lesson 15).

Every architecture in this course (GCN, GraphSAGE, GAT, R-GCN, TGN) is fundamentally a machine for producing good **node** representations; edge- and graph-level tasks are almost always built by aggregating or combining those node representations afterward.

---

# Part 1 (cont.) — Lesson 2: The Message-Passing Framework

> **This is the most important lesson — everything else in the course (GCN's matrix formula, GraphSAGE, GAT, R-GCN, TGN) is a specific instantiation of the ideas below.**

### 2.1 The big picture

Almost every modern GNN can be described as **stacking layers of local, learned "rounds of communication"** between nodes and their neighbors. This is called **Message Passing Neural Networks (MPNN)**, a unifying framework popularized by Gilmer et al. (2017) and used as the organizing structure in both reference books (Hamilton calls it the "neural message passing" framework in Ch. 5; Ma & Tang cover it as the "general GNN framework" underlying spectral and spatial convolutions in their GCN/GraphSAGE/GAT chapters).

At layer $l$, every node $v$ has a hidden state (embedding) $h_v^{(l)}$. Layer $0$ is initialized with the raw input features: $h_v^{(0)} = x_v$. One message-passing layer updates every node's state using **only its own current state and the current states of its immediate (1-hop) neighbors** $\mathcal{N}(v)$. The update happens in three conceptual steps:

$$\underbrace{\mathbf{m}_{u \to v}^{(l)} = \text{MESSAGE}^{(l)}\left(h_u^{(l)}, h_v^{(l)}, e_{uv}\right)}_{\text{1. Message}} \quad \text{for each } u \in \mathcal{N}(v)$$

$$\underbrace{\mathbf{m}_v^{(l)} = \text{AGGREGATE}^{(l)}\left(\{\mathbf{m}_{u\to v}^{(l)} : u \in \mathcal{N}(v)\}\right)}_{\text{2. Aggregation}}$$

$$\underbrace{h_v^{(l+1)} = \text{UPDATE}^{(l)}\left(h_v^{(l)}, \mathbf{m}_v^{(l)}\right)}_{\text{3. Update}}$$

Every architecture you'll learn (GCN, GraphSAGE, GAT, R-GCN, TGN) is just a different choice of the **MESSAGE**, **AGGREGATE**, and **UPDATE** functions.

### 2.2 Message

The message function decides *what information node $u$ sends to node $v$* along the edge $(u,v)$. The simplest possible choice is "just send your current hidden state": $\mathbf{m}_{u\to v} = h_u^{(l)}$. More expressive choices transform $h_u$ first (e.g., $\mathbf{m}_{u\to v} = W^{(l)} h_u^{(l)}$, as in GCN), or condition the message on edge features $e_{uv}$ (edge weight, transaction amount, relation type) or on the receiving node $h_v$ too (as attention-based methods like GAT do — the message is effectively *reweighted* by how relevant $u$ is to $v$).

Key point: the message function is the same learned function for every edge of a given type in the graph — this **parameter sharing** is what allows a GNN, unlike a shallow embedding table, to generalize to nodes/graphs it has never seen (see Lesson 5, inductive learning).

### 2.3 Aggregation

Aggregation collapses a *variable-length, unordered* set of incoming messages $\{\mathbf{m}_{u\to v} : u \in \mathcal{N}(v)\}$ into a single fixed-size vector. Because different nodes have different numbers of neighbors, and there is no natural ordering of neighbors, this function must be a **permutation-invariant set function** (see §2.4). Common choices:

- **Sum**: $\sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u\to v}$ — preserves information about neighborhood *size*/degree; most expressive in theory (this is what the GIN architecture exploits, and what the raw adjacency-matrix multiplication in Lesson 3 computes).
- **Mean**: $\frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u\to v}$ — normalizes for degree; used by GCN (via symmetric normalization) and by GraphSAGE's mean aggregator; tends to be more stable when node degrees vary a lot.
- **Max/Min-pooling**: elementwise max over neighbor messages — captures "the most extreme signal," used in some GraphSAGE variants.
- **Weighted sum (attention)**: $\sum_{u} \alpha_{uv}\, \mathbf{m}_{u\to v}$, where the weights $\alpha_{uv}$ are themselves learned/computed from the data (GAT, Lesson 6).
- **LSTM/RNN over a random neighbor ordering** — used in some GraphSAGE variants, but technically breaks strict permutation invariance unless handled carefully; included here for completeness since Hamilton discusses it as a trade-off (more expressive, order-sensitive).

### 2.4 Update

The update function combines the node's *own* previous state with the newly aggregated neighborhood message to produce the next-layer state:

$$h_v^{(l+1)} = \sigma\left(W_{\text{self}}^{(l)} h_v^{(l)} + W_{\text{neigh}}^{(l)}\, \mathbf{m}_v^{(l)}\right)$$

(a typical linear-then-nonlinearity combination; concatenation instead of addition is also common, as in GraphSAGE). $\sigma$ is a nonlinearity (ReLU, etc.). Note that the node's own previous embedding is retained as an input (often called a **self-loop** or **skip connection** at the layer level) — without this, a node's own features would get diluted or lost after a few layers of aggregation. This is conceptually the same idea as the explicit "add self-loops to $A$" trick in Lesson 3/4.

### 2.5 Permutation invariance

A node's neighborhood is an *unordered set* — there's no canonical "first neighbor" of an account. If we numbered account B's neighbors 1, 2, 3 arbitrarily and the aggregation function were sensitive to that ordering (e.g., simple concatenation), the model's output would depend on an arbitrary indexing choice, which is both meaningless and would prevent the model from generalizing (the same neighborhood presented in a different order should produce the same embedding).

Formally, AGGREGATE must satisfy, for any permutation $\pi$ of the neighbor set:

$$\text{AGGREGATE}(\mathbf{m}_1, \dots, \mathbf{m}_k) = \text{AGGREGATE}(\mathbf{m}_{\pi(1)}, \dots, \mathbf{m}_{\pi(k)})$$

Sum, mean, and max all trivially satisfy this. This is precisely why GNNs use these set functions instead of, say, concatenating neighbor features in some fixed order — and it is also the deep reason GNNs naturally handle **graphs of varying size and node degree** with one shared set of parameters, unlike a tabular model that needs a fixed number of input columns.

### 2.6 Receptive fields

The **receptive field** of node $v$ after $L$ layers of message passing is the set of nodes whose information could possibly have influenced $h_v^{(L)}$. After one layer, $v$'s embedding depends on $v$ and its 1-hop neighbors. After a second layer, it depends on $v$'s 1-hop neighbors' 1-hop neighbors too — i.e., $v$'s 2-hop neighborhood. In general:

> **After $L$ layers of message passing, node $v$'s embedding is a function of its $L$-hop neighborhood — no further, no less.**

This is a direct, mechanical consequence of stacking the message/aggregate/update triple $L$ times — exactly analogous to how stacking $L$ convolutional layers in a CNN grows the receptive field over pixels. This directly motivates:
- Why **more layers → more context**, but also more risk of *over-smoothing* (embeddings of all nodes converging to be similar once the receptive field covers most of the graph) and higher compute cost.
- Why **neighborhood explosion** (Lesson 13) is a real problem: the receptive field's *size* can grow exponentially with $L$ even though its *hop-depth* grows only linearly.

### 2.7 k-hop neighborhoods

The **k-hop neighborhood** of $v$, $\mathcal{N}_k(v)$, is the set of all nodes reachable from $v$ in exactly (or at most, depending on convention) $k$ edge traversals. Formally, $\mathcal{N}_1(v) = \mathcal{N}(v)$ (direct neighbors), and $\mathcal{N}_k(v) = \bigcup_{u \in \mathcal{N}_{k-1}(v)} \mathcal{N}(u)$.

Two practical facts you'll rely on constantly in later lessons:
1. **Depth of the GNN = size of k in "k-hop neighborhood used."** An $L$-layer GNN uses exactly the $L$-hop neighborhood — this is the direct link between architecture design (how many layers?) and how far information should realistically need to travel in your graph for the task at hand (e.g., money-laundering rings might be 3–5 hops; a fraud ring detection task might need $L=3$ or $L=4$).
2. **k-hop neighborhood size can explode combinatorially** in graphs with high average degree — this motivates neighbor *sampling* (Lesson 5, Lesson 14) rather than using the full k-hop neighborhood at scale.

---

# Part 1 (cont.) — Lesson 3: Matrix View of GNNs

Message passing can equivalently — and much more efficiently on real hardware — be expressed as a sequence of **matrix operations**. This is the bridge between the "graph diagram" intuition of Lesson 2 and the code you'll actually write in PyTorch/PyG.

### 3.1 Adjacency matrix

For a graph with $n$ nodes, the **adjacency matrix** $A \in \{0,1\}^{n\times n}$ (or $\mathbb{R}^{n \times n}$ if edges are weighted) encodes connectivity:

$$A_{ij} = \begin{cases} 1 & \text{if there is an edge from node } i \text{ to node } j \\ 0 & \text{otherwise}\end{cases}$$

For an **undirected** graph, $A$ is symmetric ($A_{ij}=A_{ji}$). For a **directed** graph (like a transaction graph, where money flows from A to B and not necessarily back), $A$ is generally *not* symmetric — and this distinction matters a lot for AML, since the direction of money flow is exactly the signal you care about.

### 3.2 Feature matrix

The **feature matrix** $H^{(l)} \in \mathbb{R}^{n \times d_l}$ (sometimes written $X$ at layer 0) stacks every node's $d_l$-dimensional hidden state as a row:

$$H^{(l)} = \begin{bmatrix} - \; h_1^{(l)\top} \; - \\ - \; h_2^{(l)\top} \; - \\ \vdots \\ - \; h_n^{(l)\top} \; - \end{bmatrix}$$

$H^{(0)} = X$ is your raw input node features (e.g., for an account node: account age, average transaction size, KYC risk score, etc.).

### 3.3 Adjacency multiplication = aggregation

This is the key insight that unlocks the whole matrix view: **left-multiplying the feature matrix by the adjacency matrix performs neighborhood sum-aggregation for every node simultaneously.**

$$(A H^{(l)})_i = \sum_{j} A_{ij}\, h_j^{(l)} = \sum_{j \in \mathcal{N}(i)} h_j^{(l)}$$

Row $i$ of $AH^{(l)}$ is exactly "sum of the feature vectors of all of $i$'s neighbors" — which is precisely the **sum-AGGREGATE** function from Lesson 2, computed for *every node in the graph in one matrix multiply*. This is why GNN layers can be implemented as (sparse) matrix multiplications and run efficiently on GPUs — there's no need to loop over nodes one at a time.

### 3.4 Self-loops

Plain $A$ has zero diagonal (a node isn't its own neighbor by default), so $AH$ **completely discards a node's own previous features** — after aggregation, $v$'s new representation only sees its neighbors, not itself. This is almost never what you want (and directly echoes the "retain self-information" point from the UPDATE step, §2.4).

The fix: add **self-loops** by defining $\hat{A} = A + I$ (where $I$ is the $n\times n$ identity matrix). Now $\hat{A}_{ii}=1$ for every node, so

$$(\hat{A} H^{(l)})_i = h_i^{(l)} + \sum_{j \in \mathcal{N}(i)} h_j^{(l)}$$

— a node's own features are automatically included in its own aggregation, in a single matrix operation, without needing a separate $W_\text{self}$ term. This is exactly the $\hat{A} = A + I$ you'll see in Lesson 4's GCN formula.

### 3.5 Degree normalization

There's a problem with using raw $\hat{A}$ (or $A$) directly: **high-degree nodes produce aggregated vectors with much larger magnitude** than low-degree nodes, purely as an artifact of summing more terms — this has nothing to do with the actual information content and destabilizes training (activations at high-degree nodes blow up; gradients become poorly scaled).

The fix is to normalize by degree. Let $D$ be the **degree matrix** — diagonal, with $D_{ii} = \sum_j A_{ij}$ (or $\hat{A}_{ij}$ if using self-loops), i.e., $D_{ii}$ = degree of node $i$. Two common normalizations:

- **Row (mean) normalization**: $D^{-1}\hat{A}$ — dividing each row by its own degree turns the sum into an *average*, matching the mean-AGGREGATE from Lesson 2. This is what GraphSAGE's mean aggregator effectively does per-neighborhood.
- **Symmetric normalization**: $D^{-1/2}\hat{A} D^{-1/2}$ — used by the original GCN (Kipf & Welling, 2017). This normalizes *both* by the source node's degree and the target node's degree, which (a) keeps the operator symmetric (important for its spectral-graph-theory derivation — this is literally a first-order approximation of spectral graph convolution, as both reference books derive in their GCN chapters) and (b) empirically trains more stably than plain row-normalization. Entry-wise:

$$\left(D^{-1/2}\hat{A}D^{-1/2}\right)_{ij} = \frac{\hat{A}_{ij}}{\sqrt{D_{ii}}\sqrt{D_{jj}}}$$

— i.e., an edge between a high-degree hub and a low-degree node gets *down-weighted* relative to an edge between two low-degree nodes, since it's divided by the (larger) square root of the hub's degree.

This exact operator, $D^{-1/2}\hat{A}D^{-1/2}$, is the "propagation matrix" you'll now derive a full GCN layer around in Lesson 4.

---

# Part 1 (cont.) — Lesson 4: GCN From Scratch

We now derive, line by line, the single most famous equation in the field — the **Graph Convolutional Network (GCN)** layer of Kipf & Welling (2017), as presented in both reference books (Hamilton §5.2–5.3 frames it as a special case of neural message passing; Ma & Tang devote a full chapter to deriving it from spectral graph convolutions before arriving at this same simplified form):

$$H^{(l+1)} = \sigma\!\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

*(Note: the equation as given in your prompt, $D^{-1/2}A D^{-1/2}H^{(l)}W^{(l)}$, is shorthand for this; in essentially every real implementation — including the original paper and PyG's `GCNConv` — $A$ here already means $\hat A = A+I$ with self-loops added, and $D$ is the degree matrix **of** $\hat A$. We make that explicit below since it's easy to silently get wrong.)*

### Line-by-line derivation

**Step 0 — start from plain message passing.** From Lesson 2, a message-passing layer is: message = neighbor's features, aggregate = normalized sum, update = linear transform + nonlinearity. GCN picks the *specific* combination:
- MESSAGE: send the raw hidden state, $\mathbf{m}_{u\to v} = h_u^{(l)}$ (transformation is deferred to *after* aggregation — a design choice, mathematically equivalent to transforming before summing since matrix multiplication is associative: $(\hat D^{-1/2}\hat A \hat D^{-1/2}) H W = (\hat D^{-1/2}\hat A \hat D^{-1/2} H)W$, but doing it after is more efficient since $W$ is applied once to an $n\times d$ matrix instead of once per edge).
- AGGREGATE: symmetric-normalized sum over $u \in \mathcal{N}(v) \cup \{v\}$ (self-loop included).
- UPDATE: linear map by $W^{(l)}$, then nonlinearity $\sigma$.

**Step 1 — add self-loops: $\hat{A} = A + I$.**
Purpose (Lesson 3.4): without this, a node's aggregated message contains *no information about itself* — after one layer, node $v$'s representation would be entirely made of its neighbors' old features, discarding $v$'s own signal completely. Adding $I$ guarantees the diagonal of $\hat A$ is 1, so every node is included in its own neighborhood sum.

**Step 2 — build the degree matrix of the self-looped graph: $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$.**
This is degree-*with*-self-loop, i.e., original degree + 1. It's needed for the next step and must be computed *after* adding self-loops, not before — a common implementation bug.

**Step 3 — symmetric normalization: $\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}$.**
Purpose (Lesson 3.5): prevents high-degree nodes from dominating purely due to magnitude, and keeps the propagation operator symmetric. Entry $(i,j)$ of this matrix equals $\frac{\hat A_{ij}}{\sqrt{\hat D_{ii}}\sqrt{\hat D_{jj}}}$ — the contribution of neighbor $j$ to node $i$ is scaled down by the *geometric mean* of both endpoints' degrees. (This specific form isn't arbitrary — it falls out of a first-order Chebyshev-polynomial approximation to spectral graph convolution using the normalized graph Laplacian $L = I - D^{-1/2}AD^{-1/2}$; both books derive this in detail if you want the full spectral motivation, but for implementation purposes the formula above is all you need.)

**Step 4 — multiply by the current-layer node features: $\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}\right) H^{(l)}$.**
This performs the normalized-sum AGGREGATE step (Lesson 3.3) for *every node simultaneously*: row $i$ of the result is a degree-normalized weighted sum of node $i$'s own (self-loop) features and all its neighbors' current features.

**Step 5 — apply the learned weight matrix: $\left(\ldots\right) H^{(l)} W^{(l)}$.**
$W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ is the (only) learned parameter of this layer — shared across *every node in the graph* (this parameter sharing is exactly what makes the GNN generalize across nodes/graphs, as discussed in Lesson 1/2). It linearly projects the aggregated $d_l$-dimensional vector into the $d_{l+1}$-dimensional space for the next layer. This is the UPDATE step's linear transform, applied to the aggregated message (with the "self" and "neighbor" transforms tied together into one $W$, unlike the separate $W_\text{self}, W_\text{neigh}$ in the generic message-passing template of §2.4 — a simplification specific to GCN).

**Step 6 — apply the nonlinearity: $\sigma(\cdot)$.**
Typically ReLU. Without this, stacking $L$ layers would collapse into a single linear map (since a composition of linear functions is linear), destroying the model's ability to represent non-linear structural patterns — exactly the same reason CNNs and MLPs need nonlinearities between layers.

**Putting it together**, one GCN layer takes $H^{(l)} \in \mathbb{R}^{n\times d_l}$ and produces $H^{(l+1)} \in \mathbb{R}^{n \times d_{l+1}}$:

$$H^{(l+1)} = \sigma\!\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

Stacking $L$ of these layers gives every node an $L$-hop receptive field (Lesson 2.6), with $H^{(0)}=X$ the raw input features and $H^{(L)}$ the final node embeddings used for downstream prediction (e.g., a softmax classifier on top for node classification).

### Minimal PyTorch (no PyG) — for intuition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, H, A_hat_norm):
        # A_hat_norm = D^{-1/2} (A + I) D^{-1/2}, precomputed once
        agg = A_hat_norm @ H          # Step 4: normalized neighbor aggregation
        out = self.W(agg)             # Step 5: learned linear transform
        return F.relu(out)            # Step 6: nonlinearity

def normalize_adj(A):
    A_hat = A + torch.eye(A.size(0))                  # Step 1: self-loops
    D_hat = torch.diag(A_hat.sum(dim=1))               # Step 2: degree matrix
    D_inv_sqrt = torch.diag(torch.pow(torch.diag(D_hat), -0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt              # Step 3: symmetric norm
```

### PyG equivalent

```python
from torch_geometric.nn import GCNConv
conv = GCNConv(in_channels=16, out_channels=32)  # handles self-loops + normalization internally
h1 = conv(x, edge_index)  # x: [n, 16], edge_index: [2, num_edges]
```

`GCNConv` performs *exactly* steps 1–6 above internally, given `edge_index` (PyG's sparse edge-list representation of $A$) — this is why understanding the derivation matters even though you'll rarely hand-write it in production.

---

# Part 2 — Important Architectures

## Lesson 5 — GraphSAGE

GraphSAGE ("SAmple and aggreGatE," Hamilton, Ying & Leskovec, 2017) was designed to solve two problems GCN has out of the box: it doesn't scale to huge graphs, and (in its original formulation) it's **transductive** — it can't produce embeddings for nodes that weren't present at training time.

### 5.1 Neighbor sampling

Instead of aggregating over a node's *entire* neighborhood (which for a hub node in a large transaction/social graph could be thousands of nodes), GraphSAGE **randomly samples a fixed-size subset** of neighbors at each layer, e.g., sample up to 25 neighbors at layer 1 and up to 10 at layer 2 (numbers used in the original paper). This bounds the per-node compute cost at every layer to a constant, regardless of how skewed the degree distribution is — directly addressing the neighborhood-explosion problem you'll formalize in Lesson 13.

### 5.2 Mean aggregator

GraphSAGE proposes a family of AGGREGATE functions (mean, LSTM, pooling); the **mean aggregator** is the simplest and most widely used:

$$h_{\mathcal{N}(v)}^{(l)} = \text{MEAN}\left(\{h_u^{(l-1)} : u \in \mathcal{N}_{\text{sample}}(v)\}\right)$$
$$h_v^{(l)} = \sigma\left(W^{(l)} \cdot \text{CONCAT}\left(h_v^{(l-1)},\ h_{\mathcal{N}(v)}^{(l)}\right)\right)$$

Two design differences from GCN worth noting explicitly: (1) the node's own previous embedding $h_v^{(l-1)}$ is **concatenated** with (not summed into) the neighbor aggregate, keeping "self" and "neighbor" information more distinctly separated through the transform; (2) aggregation is over the *sampled* neighbor set, not the full neighborhood.

### 5.3 Inductive learning

**Transductive** models (plain shallow embeddings, and GCN in its original per-graph formulation) learn a fixed lookup table / operate over one fixed, complete adjacency matrix seen at training time — they have no defined way to produce an embedding for a node that didn't exist in that matrix during training. If a new account opens tomorrow, a purely transductive pipeline needs to be *completely retrained*.

**Inductive** models learn a *function* — MESSAGE/AGGREGATE/UPDATE — that only depends on (sampled) local neighborhood structure and node features, not on a fixed global node index. GraphSAGE is inductive by construction: because it never learns a per-node embedding table, and always aggregates over *whatever* neighbors a node currently has, it can be applied directly to brand-new nodes (or entirely new graphs) at inference time without retraining. This is exactly why GraphSAGE (and GCN too, when implemented in the "message passing" style rather than the original full-graph spectral style, which is how PyG's `SAGEConv`/`GCNConv` are actually implemented) is the practical default for production settings like AML, where new accounts and transactions appear continuously.

### 5.4 Large graphs

Putting sampling + inductive learning + mean aggregation together is what makes GraphSAGE the standard choice for **large graphs**: (a) sampling bounds compute per node so you never need the full adjacency matrix in memory or need to compute over the full neighborhood of a hub, (b) mini-batches of just the nodes you care about (plus their sampled k-hop neighborhoods) can be trained on GPU without loading the whole graph, and (c) inductive generalization means you don't have to retrain from scratch as the graph grows. This is the direct technical foundation for PyG's `NeighborLoader` (Lesson 14).

---

## Lesson 6 — GAT (Graph Attention Networks)

### 6.1 Attention

GCN and GraphSAGE-mean both treat all neighbors as (after normalization) roughly equally important, weighted only by fixed, structural quantities (degree). But intuitively, **not all neighbors should count equally** — for an account under investigation, a transaction to/from a known shell company should matter more than a transaction to/from a well-established retailer, even if both are just "one edge." **Graph Attention Networks** (Veličković et al., 2018) let the model *learn* how much to weight each neighbor, conditioned on the actual features of both nodes — this makes the AGGREGATE step from Lesson 2 a learned, data-dependent weighted sum instead of a fixed-formula one.

### 6.2 Attention coefficients

For an edge $(u, v)$ (message flowing from neighbor $u$ into target $v$), GAT computes a raw attention score using a shared linear transform $W$ and a learned attention vector $\mathbf{a}$:

$$e_{uv} = \text{LeakyReLU}\left(\mathbf{a}^\top \left[W h_u \,\Vert\, W h_v\right]\right)$$

($\Vert$ = concatenation.) This score is then normalized across *all* of $v$'s neighbors using softmax, so the final attention coefficients sum to 1 and are comparable across neighborhoods of different sizes:

$$\alpha_{uv} = \text{softmax}_u(e_{uv}) = \frac{\exp(e_{uv})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{kv})}$$

The AGGREGATE + UPDATE steps then become a weighted sum using these learned weights:

$$h_v^{(l+1)} = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{uv}\, W h_u^{(l)}\right)$$

Crucially, $\alpha_{uv}$ depends on the *features* of both $u$ and $v$ at the current layer — so the same node $u$ can be weighted very differently for two different target nodes it's connected to, and the weighting adapts as embeddings update across layers. This also still satisfies permutation invariance (Lesson 2.5): softmax over a set, and weighted-sum aggregation, don't depend on any ordering of the neighbors.

### 6.3 Multi-head attention

Just like Transformers, GAT typically computes **several independent attention mechanisms in parallel** ("heads"), each with its own $W^{(k)}, \mathbf{a}^{(k)}$, and combines the results — concatenation for intermediate layers, averaging for the final layer:

$$h_v^{(l+1)} = \Big\Vert_{k=1}^{K} \sigma\left(\sum_{u\in\mathcal{N}(v)} \alpha_{uv}^{(k)} W^{(k)} h_u^{(l)}\right) \quad \text{(intermediate layers)}$$

$$h_v^{(L)} = \sigma\left(\frac{1}{K}\sum_{k=1}^{K}\sum_{u\in\mathcal{N}(v)} \alpha_{uv}^{(k)} W^{(k)} h_u^{(l)}\right) \quad \text{(final layer)}$$

Multiple heads let the model learn **different "types" of relevance** simultaneously — one head might learn to attend to high-value transactions, another to recency, another to structural role — analogous to how multi-head self-attention in a Transformer lets different heads specialize in different relationships between tokens. This added expressiveness comes at the cost of more parameters and compute than GCN/GraphSAGE.

---

## Lesson 7 — Comparing GCN, GraphSAGE, and GAT

| Model | Main idea | Why useful |
|---|---|---|
| **GCN** | Normalized neighbor aggregation — degree-based symmetric normalization $\hat D^{-1/2}\hat A\hat D^{-1/2}$, same fixed weighting rule for every edge | Simple, cheap, strong baseline; theoretically grounded in spectral graph theory; the standard first thing to try on a new graph task |
| **GraphSAGE** | Sampled neighbor aggregation — randomly subsample a fixed number of neighbors per layer, mean (or pool/LSTM) aggregate, concatenate with self | Scalable to large graphs (bounded per-node compute); inductive — generalizes to unseen nodes/graphs without retraining; ideal for production and streaming data |
| **GAT** | Weighted neighbor aggregation — attention coefficients learned from node-pair features, softmax-normalized per neighborhood, multi-head | Learns *which* neighbors matter instead of assuming fixed structural weighting; more expressive, better when neighbor importance varies a lot (e.g., a mostly-legitimate account with one suspicious counterparty) |

**When to reach for which, in practice:**
- Start with **GCN** as a baseline whenever the graph is small/medium and fully available at training time (transductive is fine) — it's fast, simple, and a reasonable floor on performance.
- Move to **GraphSAGE** as soon as (a) the graph is too large for full-batch training, (b) new nodes/edges arrive continuously (streaming/production setting), or (c) you need mini-batch training on GPU-memory-constrained hardware.
- Move to **GAT** (or add attention on top of a GraphSAGE-style sampler — this combination is common in practice) when you suspect neighbor importance is highly heterogeneous — e.g., in fraud/AML, where 99% of an account's counterparties are benign and 1% are the signal you actually care about, uniform or degree-based weighting can wash out the important edges, while attention can learn to spotlight them.

All three are message-passing instances (Lesson 2) that differ only in their choice of MESSAGE/AGGREGATE, and all three can be expressed via `torch_geometric.nn.{GCNConv, SAGEConv, GATConv}` with a nearly identical calling convention, which is why swapping architectures in PyG is usually a one-line change.

---

# Part 3 — Heterogeneous GNNs

## Lesson 8 — Homogeneous vs. Heterogeneous Graphs

### 8.1 Homogeneous vs. heterogeneous

A **homogeneous graph** has exactly one node type and one edge type — everything discussed in Lessons 2–7 implicitly assumed this (e.g., "account → account" transactions only, all accounts treated identically). A **heterogeneous graph** has **multiple node types and/or multiple edge (relation) types** — e.g., `Account`, `Bank`, `Transaction`, `Currency` as different node types (exactly the setup previewed in Lesson 15's "Option C").

The reason this distinction matters architecturally: a single shared weight matrix $W$ (as in plain GCN) implicitly assumes every node/edge means "the same kind of thing." Applying the same $W$ to an `Account` node's features and a `Bank` node's features (which may not even share the same feature schema — an account has a "risk score," a bank has a "regulatory jurisdiction") is semantically meaningless. Heterogeneous GNNs (Lesson 9) fix this by learning **separate parameters per type**.

### 8.2 Node types and edge types

- **Node type**: a label indicating what *kind* of entity a node represents (`Account`, `Bank`, `Transaction`, `Currency`, ...). Each node type can have its own feature schema/dimensionality.
- **Edge type**: a label indicating what *kind* of relation an edge represents (`owns`, `sends_to`, `denominated_in`, `hosted_by`, ...). Different edge types can carry entirely different semantics even if they connect the same pair of node types (e.g., `Account`–`Account` could have both a `sends_money_to` and a `shares_address_with` relation, which mean very different things for AML).

### 8.3 Canonical edge types

Because the *same pair of node types* can be connected by *multiple, semantically distinct* edge types, PyG (and heterogeneous graph literature generally) identifies a relation not just by its name but by the full triple:

$$(\text{source node type},\ \text{relation name},\ \text{destination node type})$$

e.g. `('Account', 'sends_to', 'Account')`, `('Account', 'owns', 'Bank')`, `('Transaction', 'denominated_in', 'Currency')`. This triple is called a **canonical edge type**, and it's the actual key used to look up which learned parameters (which relation-specific weight matrix, per Lesson 9) apply to a given edge.

### 8.4 PyG `HeteroData`

`torch_geometric.data.HeteroData` is PyG's container for heterogeneous graphs: it stores a separate feature matrix per node type, and a separate `edge_index` per canonical edge type.

```python
from torch_geometric.data import HeteroData

data = HeteroData()
data['account'].x = account_features          # [num_accounts, d_account]
data['bank'].x = bank_features                 # [num_banks, d_bank]
data['transaction'].x = transaction_features    # [num_transactions, d_txn]

data['account', 'sends_to', 'account'].edge_index = send_edge_index
data['account', 'owns', 'bank'].edge_index = owns_edge_index
data['account', 'initiates', 'transaction'].edge_index = init_edge_index
data['transaction', 'received_by', 'account'].edge_index = recv_edge_index
```

Every downstream tool (loaders, `HeteroConv`, `to_hetero()`) operates on this canonical-edge-type structure, which is why getting the schema right at this stage (directly informed by the Lesson 15 graph-formulation decision) matters so much for everything that follows.

---

## Lesson 9 — R-GCN and Heterogeneous Message Passing

### 9.1 R-GCN: relation-specific weights

**Relational GCN** (Schlichtkrull et al., 2018) extends the GCN update rule (Lesson 4) to heterogeneous graphs by giving **each relation type its own weight matrix** $W_r$, so the model can learn that "money flowing in via a `sends_to` edge" should be processed completely differently from "ownership via an `owns` edge":

$$h_v^{(l+1)} = \sigma\left(W_0^{(l)} h_v^{(l)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{c_{v,r}}\, W_r^{(l)} h_u^{(l)}\right)$$

where $\mathcal{R}$ is the set of relation types, $\mathcal{N}_r(v)$ is $v$'s neighbors connected via relation $r$ specifically, $W_0$ is a self-loop transform (same idea as Lesson 3.4/4's self-loop, but kept separate here rather than folded into $\hat A$), and $c_{v,r}$ is a normalization constant (e.g., $|\mathcal{N}_r(v)|$, analogous to the degree normalization of Lesson 3.5, but computed *per relation type*).

**Practical caveat directly relevant to AML** (this is explicitly discussed as a limitation in the R-GCN paper): with many relation types, the number of parameters ($|\mathcal{R}|$ separate $W_r$ matrices) can grow very large and overfit, especially on relation types with few examples. R-GCN addresses this with **basis decomposition** — expressing every $W_r$ as a shared linear combination of a small number of shared "basis" matrices, $W_r = \sum_{b=1}^{B} a_{rb} V_b$, drastically reducing the parameter count while still letting each relation learn its own combination.

### 9.2 `HeteroConv` and type-specific transformations

PyG operationalizes this relation-specific idea via `torch_geometric.nn.HeteroConv`, which wraps a *dictionary of ordinary (homogeneous) convolutions*, one per canonical edge type, and sums (or means/concatenates) their outputs into each node type's next-layer representation:

```python
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear

conv = HeteroConv({
    ('account', 'sends_to', 'account'): SAGEConv((-1, -1), 64),
    ('account', 'owns', 'bank'):         SAGEConv((-1, -1), 64),
    ('account', 'initiates', 'transaction'): GATConv((-1, -1), 64, add_self_loops=False),
    ('transaction', 'received_by', 'account'): GATConv((-1, -1), 64, add_self_loops=False),
}, aggr='sum')

out_dict = conv(x_dict, edge_index_dict)   # x_dict, out_dict: {'account': tensor, 'bank': tensor, 'transaction': tensor}
```

Each entry is a *type-specific transformation*: a completely separate learned convolution (its own weights) for that particular canonical edge type — this is the direct PyG realization of R-GCN's $W_r$-per-relation idea (and, because you can mix conv *types* per relation, e.g. `SAGEConv` for one relation and `GATConv` for another, it's actually more flexible than the original R-GCN paper). PyG also offers `to_hetero(model, metadata)`, which automatically converts an *existing homogeneous* model definition into a heterogeneous one by replicating it per edge type — useful when you've already built and debugged a homogeneous GCN/SAGE/GAT model and want to "upgrade" it without rewriting the forward pass.

---

# Part 4 — Temporal GNNs

## Lesson 10 — Static vs. Dynamic Graphs

### 10.1 Static vs. dynamic

Everything up through Lesson 9 assumed a **static graph** — one fixed $A$, $X$ that doesn't change. Real transaction data is inherently **dynamic**: new accounts open, new transactions occur continuously with real timestamps, and *the order and timing of events carries signal* (e.g., a burst of small transactions in a short window — "smurfing" — is a temporal pattern, not just a structural one; a static graph snapshot would show the same edges whether they occurred over 10 minutes or 10 months).

### 10.2 Snapshots vs. event-based (continuous-time) graphs

Two common ways to model a dynamic graph:

- **Snapshot-based (discrete-time)**: the timeline is chopped into fixed windows (e.g., one graph per day), producing a *sequence* of static graphs $G_1, G_2, \dots, G_T$. You then apply a GNN to each snapshot and a sequence model (RNN/Transformer) across snapshots. Simple to implement (you can mostly reuse Lessons 2–9's machinery per snapshot) but loses fine-grained ordering *within* a window and requires choosing an arbitrary window size — a bad choice can either smear together a fast-moving laundering pattern (window too coarse) or fragment a slower one across too many empty snapshots (window too fine).
- **Event-based (continuous-time)**: every edge (transaction) carries its own exact timestamp, and the model processes events as a *stream*, updating node representations incrementally as each event arrives, rather than in fixed batches. This is more faithful to real transaction data (which has no natural "window" boundary) and avoids the window-size hyperparameter — at the cost of more complex architectures (this is the regime TGNs, Lesson 12, operate in).

### 10.3 Timestamps

In an event-based graph, each edge $(u, v, t, \text{features})$ carries a timestamp $t$ (and possibly edge features like transaction amount). This timestamp is what distinguishes "A sent money to B, then B sent money to C" (a plausible 2-hop laundering chain, since B *received before* it *sent*) from "B sent money to C, then A sent money to B" (not a valid causal chain — B couldn't have forwarded money it hadn't received yet). Encoding and respecting this ordering is the whole point of temporal GNNs.

### 10.4 Temporal neighbors

In a static graph, "$v$'s neighbors" is just $\mathcal{N}(v)$. In a temporal graph, we usually need the **temporal (or "history-respecting") neighborhood as of time $t$**:

$$\mathcal{N}(v, t) = \{u : (u,v,t') \in E,\ t' < t\}$$

i.e., only neighbors connected to $v$ *before* the query time $t$ count — this prevents the model from "looking into the future" when computing a representation meant to be used for a prediction at time $t$ (directly foreshadowing "temporal leakage" in Lesson 11). This also means the same node can have a *different* temporal neighborhood (and thus different embedding) depending on *when* you query it — a fundamentally different object from the single fixed embedding a static GNN produces.

---

## Lesson 11 — Temporal Message Passing

### 11.1 Time encoding

To let a neural network use a raw timestamp (or more usefully, a *time gap* like "14 minutes since the previous transaction") as an input, it needs to be turned into a vector, just like positional encoding in a Transformer. A common approach (used in TGAT, Xu et al. 2020, and adopted by TGN) is a **learnable functional time encoding** based on Bochner's theorem / Fourier features:

$$\Phi(\Delta t) = \left[\cos(\omega_1 \Delta t), \cos(\omega_2 \Delta t), \dots, \cos(\omega_d \Delta t)\right]$$

with $\omega_i$ learned parameters. This gives the model a continuous, differentiable representation of "how long ago did this happen," which can then be concatenated onto node/edge features and fed through ordinary message-passing machinery — turning "time" from a hard-to-use scalar into a rich vector the rest of the network can combine with feature information (e.g., attention over neighbors, as in GAT, can now be conditioned on *both* feature similarity *and* recency).

### 11.2 Temporal message passing

Temporal message passing is the same MESSAGE/AGGREGATE/UPDATE template from Lesson 2, but every message now additionally depends on *when* the interaction happened, and aggregation is restricted to the temporal neighborhood as of the query time:

$$\mathbf{m}_{u\to v}(t) = \text{MESSAGE}\left(h_u(t_u^-),\ h_v(t^-),\ e_{uv},\ \Phi(t - t_{uv})\right), \quad u \in \mathcal{N}(v, t)$$

where $t_u^-$ denotes "the most recent time $u$'s state was updated, before $t$" and $\Phi$ is the time encoding above. The key architectural difference from a static GNN: the same node's embedding is not a single fixed vector but a **function of query time**, $h_v(t)$, and computing it requires walking back through $v$'s (and its neighbors') interaction history up to (but not past) $t$.

### 11.3 Causal temporal aggregation

**Causal** aggregation means the representation used for node $v$ at time $t$ is computed *only* from events at times $< t$ (or $\le t$ depending on convention, but always excluding the future). This is not just good practice, it's what makes the model's predictions actually deployable: at inference time, when you're scoring a transaction as it happens, you genuinely *do not have access* to future transactions — so if training doesn't respect that same constraint, the model will have learned to rely on information it will never have at serving time.

### 11.4 Temporal leakage

**Temporal leakage** is the (extremely common, extremely damaging) bug where, despite intending causal aggregation, some future information leaks into a node's training-time representation — typically because of one of:
- Building the graph/features **once, globally**, over the entire dataset (e.g., computing "total transaction volume for this account" using all-time data, including transactions that happen *after* the labeled event you're training on).
- Doing **random train/val/test splits** over transactions/accounts instead of **time-based splits** — a random split lets a model implicitly "see" temporally-later transactions of an account (via message passing or feature engineering) while being evaluated on temporally-earlier ones for the same account.
- Batching by snapshot but allowing message passing *within* a batch to flow from a later timestamp to an earlier one in the same window (this can happen if edges within a snapshot aren't causally ordered during aggregation).

Temporal leakage produces models that look great in offline validation and then perform far worse in production, because the "signal" they were actually relying on (future information) simply doesn't exist yet at real serving time. For an AML system specifically, this is the single most important methodological pitfall to check for, and it's why Lesson 12+ builds toward architectures (TGN) and evaluation protocols designed around strictly chronological splits.

---

## Lesson 12 — Temporal Graph Networks (TGN)

### 12.1 Memory-based approaches

**Temporal Graph Networks** (Rossi et al., 2020) address a practical problem with the "TGAT-style" pure temporal-attention approach of Lesson 11: recomputing a node's embedding from scratch by walking back through its *entire* temporal neighborhood every time you need it is expensive, and doesn't naturally carry forward a compact "summary" of everything relevant that's happened to a node so far.

TGN's solution: give **every node a memory vector** $s_v(t)$ — a compact, continuously-updated state that summarizes the node's interaction history up to time $t$, conceptually similar to an RNN's hidden state, but updated *event-by-event* (asynchronously, per node, whenever that specific node is involved in an interaction) rather than at fixed time steps. The architecture has a few key modules:

1. **Message function**: when an event $(u, v, t, e_{uv})$ occurs, compute messages for both endpoints, e.g. $\mathbf{msg}_u = \text{MSG}(s_u(t^-), s_v(t^-), \Delta t, e_{uv})$ (and symmetrically for $v$).
2. **Memory updater** (typically a GRU/RNN cell): $s_v(t) = \text{GRU}(\mathbf{msg}_v, s_v(t^-))$ — updates the node's memory using the new message and its previous memory.
3. **Embedding module**: the actual node embedding used for downstream prediction, $z_v(t)$, is computed by combining the memory $s_v(t)$ with a (typically shallow, 1–2 hop) graph-attention pass over recent temporal neighbors (i.e., the memory acts as a compressed long-range summary, while a small amount of explicit message passing over recent neighbors adds fine-grained local/current context).

This memory mechanism is what lets TGN scale to long interaction histories without needing to recompute deep temporal neighborhoods from scratch at every query, while still respecting strict causal ordering (Lesson 11.3) since the memory for $v$ is only ever updated using events up to the current time.

### 12.2 Sequential transaction modeling

For AML specifically, TGN's memory mechanism maps very naturally onto **sequential transaction modeling**: an account's memory vector is effectively a running, learned summary of "everything relevant about this account's behavior so far" — updated incrementally as each new transaction arrives, exactly matching how a real-time AML monitoring system needs to operate (score each transaction as it streams in, using only past context, without recomputing the whole graph from scratch). This is in contrast to snapshot-based approaches (Lesson 10.2), which would need to re-run a full GNN pass over a fresh graph snapshot to get updated account states.

> **On "which temporal architecture actually fits the IBM AML setup":** with the full toolkit above in place — snapshot vs. event-based (10.2), causal temporal neighborhoods (10.4), time encoding (11.1), causal aggregation and leakage (11.3–11.4), and memory-based sequential modeling (12.1–12.2) — the actual choice for a specific dataset (e.g., the IBM AML synthetic transaction dataset) depends on concrete properties of that dataset: how many events per account, whether transactions arrive as a continuous stream or naturally batch into time windows, how far back "relevant history" typically needs to reach, and your latency/serving constraints. That's a data- and requirements-driven decision to work through explicitly against the dataset (e.g., examining its time span, event density, and label structure) rather than something to resolve generically in notes — happy to work through that comparison concretely once you're ready to look at the dataset's actual statistics.

---

# Part 5 — Scaling

## Lesson 13 — Why Full-Batch GNN Training Fails at Scale

### 13.1 Computation graphs

To compute the loss gradient for even a *single* labeled node $v$ using an $L$-layer GNN, autograd must build a computation graph that includes: $v$'s layer-$L$ computation, which depends on $v$'s $(L{-}1)$-layer neighbors, each of which depends on *their* $(L{-}2)$-layer neighbors, and so on, down to layer 0. In full-batch training (the "textbook" way to train GCN, matching Lesson 4's matrix formula literally), this computation graph is built over the **entire graph at once** — every node's representation at every layer, for every node in the training set, all held in memory simultaneously for the backward pass.

### 13.2 Neighborhood explosion

The problem: the number of nodes pulled into a single target node's computation graph doesn't grow *linearly* with the number of layers $L$ — it grows roughly **multiplicatively** with the average degree, since each hop multiplies by (roughly) the average branching factor. Concretely, as given in the lesson prompt:

```
100 neighbors
     ↓
100 × 100 after 2 hops
     ↓
10,000 potential nodes
```

With a 3rd layer, that could balloon to ~1,000,000 nodes for a *single* target node's receptive field — even though the graph itself might "only" have a few million nodes total, meaning a handful of target nodes' computation graphs could already cover almost the entire graph. This is a direct, unavoidable consequence of the receptive-field growth described in Lesson 2.6 ("after $L$ layers, node $v$'s embedding depends on its $L$-hop neighborhood") combined with real-world degree distributions, which are rarely uniform — a small number of hub nodes (e.g., a payment processor or exchange account in a transaction graph) can have enormous degree, and if such a hub falls within a target node's $L$-hop neighborhood, it single-handedly explodes the computation graph size.

**Consequences in practice**: full-batch training on a graph with millions of nodes either (a) doesn't fit in GPU memory at all, or (b) makes even a single training step extremely slow because of redundant, massively-overlapping computation across nearby target nodes. This motivates the sampling-based mini-batching strategies of Lesson 14, which trade a controlled amount of variance/approximation for a **hard, predictable cap** on the size of every node's receptive field, regardless of the graph's actual degree distribution.

---

## Lesson 14 — Neighbor Sampling, Mini-Batching, and Subgraph Extraction

### 14.1 Neighbor sampling

As introduced conceptually in GraphSAGE (Lesson 5.1), **neighbor sampling** caps the receptive-field explosion (Lesson 13.2) by randomly selecting a *fixed maximum number* of neighbors to aggregate over at each layer, instead of using all of them. E.g., "sample at most 15 neighbors at hop 1, at most 10 at hop 2" bounds a 2-layer model's receptive field to at most $15 \times 10 = 150$ nodes per target node, regardless of the true degree of any node involved — turning an unbounded, degree-dependent computation graph into a bounded, predictable one.

### 14.2 Mini-batching

Rather than computing embeddings for every node in the graph in one pass, **mini-batch training** picks a small batch of *target* nodes (the ones you actually have labels/loss for in this step), and constructs — via neighbor sampling — just enough of the surrounding graph (their sampled $L$-hop neighborhoods) to compute those target nodes' embeddings and gradients. This mirrors ordinary mini-batch SGD from standard deep learning, but with the added twist that each "example" (target node) needs its own *subgraph* of dependencies, not just an isolated feature vector.

### 14.3 Subgraph extraction

The practical mechanism underlying both of the above: for a given mini-batch of target (or "seed") nodes, the loader **extracts a subgraph** consisting of the seed nodes plus their sampled multi-hop neighbors (and the edges connecting them), relabels these to a small local index space, and hands this compact subgraph — not the full graph — to the GNN for a forward/backward pass. Since this subgraph is typically orders of magnitude smaller than the full graph, it fits comfortably in GPU memory and trains quickly, and because the sampling is stochastic and re-drawn every epoch, the model still sees a representative, varied sample of each node's neighborhood over the course of training.

### 14.4 `NeighborLoader`

`torch_geometric.loader.NeighborLoader` implements exactly this pipeline for node-level tasks: given a full (potentially huge) graph stored once in memory/on disk, a set of seed nodes, and a `num_neighbors` list specifying how many neighbors to sample per hop, it yields mini-batch subgraphs on the fly.

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,                          # full (homogeneous or Hetero) graph
    num_neighbors=[15, 10],        # sample 15 neighbors at hop 1, 10 at hop 2 (2-layer model)
    batch_size=256,                # 256 seed/target nodes per mini-batch
    input_nodes=train_mask,        # only sample seeds from the training set
    shuffle=True,
)

for batch in loader:
    out = model(batch.x, batch.edge_index)
    loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])  # loss only on seed nodes
    loss.backward()
    optimizer.step()
```

Note the important convention: PyG places the seed/target nodes first in the returned batch (`batch.batch_size` of them), with their sampled neighbors appended after — so the loss is computed only on the first `batch_size` rows even though the batch contains many more (neighbor) nodes needed purely to *compute* those target embeddings.

### 14.5 `LinkNeighborLoader`

For **edge/link-level tasks** (link prediction, or — directly relevant to Lesson 16 — transaction classification when a transaction is modeled as an edge, Lesson 15 Option A), the analogous tool is `torch_geometric.loader.LinkNeighborLoader`. Instead of sampling neighborhoods around *seed nodes*, it samples neighborhoods around the **two endpoint nodes of seed edges**, and (crucially, for link prediction) can automatically generate negative edges (non-existent edges) for contrastive training:

```python
from torch_geometric.loader import LinkNeighborLoader

loader = LinkNeighborLoader(
    data,
    num_neighbors=[15, 10],
    edge_label_index=train_edge_index,   # the (positive) edges to predict/classify
    edge_label=train_edge_label,          # e.g., 1 = fraudulent txn, 0 = legitimate
    batch_size=256,
    neg_sampling_ratio=1.0,               # only relevant for unsupervised link prediction, not needed if you already have labeled negatives
    shuffle=True,
)
```

This is the loader you'd reach for directly once Lesson 15/16 settle on "transaction = edge, and we're classifying transactions" as the problem formulation.

---

# Part 6 — AML-Specific Learning

## Lesson 15 — Turning Transactions Into a Graph

> This is one of the most important project-design decisions in the whole course — it determines every architectural choice downstream (which GNN layer types apply, which loader you use, what the prediction target even *is* in Lesson 16).

### Option A — Account →(transaction)→ Account; transaction = edge

**Structure**: nodes are `Account` only; each transaction becomes a *directed, weighted, timestamped edge* from the sender account to the receiver account (edge features: amount, currency, timestamp, transaction type, etc.).

- **Pros**: simplest possible graph — homogeneous (Lesson 8 doesn't even apply), directly compatible with plain GCN/GraphSAGE/GAT (Lessons 4–7) with edge features folded into the MESSAGE function; very memory-efficient (number of edges = number of transactions, no extra transaction nodes); natural fit if the prediction target is **account-level** risk (Lesson 16).
- **Cons**: two accounts can transact multiple times, so this is naturally a **multigraph** (multiple parallel edges between the same pair) — many GNN layers/implementations assume simple graphs and need adaptation (e.g., aggregate multi-edges' features before/during message passing, or use frameworks that natively support multigraphs); harder to directly predict something *about a specific transaction* (since a transaction isn't a node, you don't get a first-class transaction embedding — you'd have to build one from the two endpoint account embeddings plus the edge features, e.g. via an edge-level decoder as in Lesson 1.4).

### Option B — Account → Transaction ← Account; transaction = node

**Structure**: two node types, `Account` and `Transaction`; each transaction becomes its own node, connected via two (typed) edges: `Account —initiates→ Transaction` and `Transaction —received_by→ Account` (this is exactly a **heterogeneous graph**, Lesson 8, in miniature).

- **Pros**: transactions get a **first-class node representation** $z_{txn}$, which is exactly what you want if the prediction target is **transaction classification** (Lesson 16) — you can attach transaction features directly to the transaction node and read off its embedding directly, no need to combine two account embeddings after the fact; naturally handles multiple transactions between the same account pair without any multigraph complications (each is simply a separate node); can attach rich transaction-specific features (amount, timestamp, channel, currency) as that node's own feature vector rather than squeezing them into an edge-feature vector.
- **Cons**: doubles-plus the node count (every transaction is now a node), increasing memory/compute; requires heterogeneous machinery (`HeteroData`, `HeteroConv`/R-GCN, Lesson 8–9) even for this fairly simple two-type case; a 1-hop-per-relation "account→transaction→account" path means you typically need **at least 2 GNN layers** just to let information flow from one account, through a transaction, to the other account — effectively halving your "real" structural depth for the same layer count compared to Option A.

### Option C — Richer heterogeneous graph: Account, Bank, Transaction, Currency, multiple relation types

**Structure**: extends Option B with additional node types (`Bank`, `Currency`, possibly `Country`, `Person`/beneficial-owner, `Device`, etc.) and additional relation types (`Account —owns_by→ Person`, `Account —hosted_at→ Bank`, `Transaction —denominated_in→ Currency`, `Bank —located_in→ Country`, ...) — this is the full realization of the heterogeneous machinery from Lessons 8–9.

- **Pros**: captures the *richest* set of real-world relational signals — e.g., "many shell accounts hosted at the same small offshore bank" or "a cluster of accounts sharing a beneficial owner" are exactly the kinds of multi-entity patterns real AML typologies (layering, smurfing, shell-company networks) rely on, and which Options A/B structurally cannot represent at all since they only have `Account` (and `Transaction`) nodes; supports **subgraph/ring-level classification** (Lesson 16) naturally, since a "ring" is most faithfully represented as a connected subgraph spanning multiple entity types.
- **Cons**: substantially more complex to build and maintain (schema design, data engineering to populate every node/edge type, keeping it updated as new entity types are onboarded); needs R-GCN/`HeteroConv` (Lesson 9) with per-relation weights, which means more parameters and a higher risk of overfitting relation types with sparse data (mitigated by basis decomposition, Lesson 9.1); heavier compute and memory footprint than A or B.

### Choosing among them

There isn't a universally "correct" option — the right choice is driven directly by **what you decide the prediction target is** (Lesson 16) and what data is actually available:
- If the target is **account risk classification** and you don't have rich auxiliary entities (banks, currencies, etc.) readily available → **Option A** is simplest and sufficient.
- If the target is **transaction classification** (flag *this specific transfer* as suspicious) → **Option B** gives transactions a first-class representation, which is the more natural fit.
- If the target is **ring/typology detection** (a *group* of accounts + transactions forming a laundering pattern) and you have auxiliary entity data (shared banks, shared beneficial owners, shared devices, etc.) → **Option C** is worth the added complexity, since it's the only formulation that can structurally represent the multi-entity relationships those typologies actually depend on.

Working through this against the *specific* dataset you have (e.g., the IBM AML synthetic dataset's actual schema — what entities/fields it provides) is exactly the right next step once you're ready to move from these notes into implementation.

---

## Lesson 16 — What Exactly Is the Prediction Target?

The choice of prediction target is not a detail decided *after* modeling — Hamilton and Ma & Tang both frame the "decoder" (Lesson 1.1) as something chosen jointly with the encoder specifically because the target determines the loader, decoder, loss, and evaluation protocol all at once, as this lesson lays out explicitly.

### 16.1 Transaction classification

**Target**: is this specific transaction suspicious/laundering-related? A **binary (or multi-class, if labeling by typology) label per transaction/edge**.
- **Loader**: `LinkNeighborLoader` (Lesson 14.5) if transaction = edge (Option A); or `NeighborLoader` seeded on `Transaction`-type nodes (Lesson 14.4, with `input_nodes=('transaction', train_mask)`) if transaction = node (Option B).
- **Decoder**: for edge-as-transaction (Option A), an MLP over the concatenation/Hadamard product of the two endpoint account embeddings (+ edge features): $\hat{y}_{uv} = \text{MLP}([z_u \Vert z_v \Vert e_{uv}])$. For node-as-transaction (Option B), a simple MLP classifier head directly on the transaction node's own embedding: $\hat{y}_{txn} = \text{MLP}(z_{txn})$.
- **Loss**: binary cross-entropy (or weighted/focal, Lesson 17, given the extreme class imbalance typical of AML labels).
- **Evaluation**: PR-AUC / F1 at the transaction level (Lesson 17).

### 16.2 Account classification

**Target**: is this account itself risky/complicit (regardless of any single transaction)? A **label per account/node**.
- **Loader**: `NeighborLoader` seeded on `Account`-type nodes.
- **Decoder**: a classifier head directly on the account node's embedding $z_v$ — the standard node-classification setup underlying essentially every example in Lessons 4–9.
- **Loss/Evaluation**: same considerations as above, at the account level instead of transaction level.

### 16.3 Subgraph / ring classification

**Target**: does this *group* of accounts/transactions constitute a laundering ring/typology (e.g., a smurfing structure, a circular flow, a layering chain)? A **label per subgraph** (a candidate ring, however it was identified — e.g., via community detection, a fixed-size ego-network around a flagged seed account, or a connected component of recent high-risk transactions).
- **Loader**: requires extracting candidate subgraphs first (this is a data-engineering step *before* the GNN loader — e.g., building an ego-network of radius $k$ around each candidate seed, or a k-hop subgraph extractor), then batching these subgraphs (PyG's standard `DataLoader` over a list of `Data`/`HeteroData` subgraph objects, using PyG's automatic graph-batching-via-block-diagonal-adjacency mechanism).
- **Decoder**: requires a **readout/pooling** step (Lesson 1.4) — global mean/sum/max pooling, or a learned attention-pooling — over all node embeddings in the candidate subgraph, producing one graph-level vector $z_G$, then an MLP classifier on top of $z_G$.
- **Loss/Evaluation**: same imbalance-aware losses/metrics (Lesson 17), now at the subgraph/ring level; note that generating good **negative** (non-ring) candidate subgraphs for training is itself a nontrivial design problem (e.g., random subgraphs of similar size vs. "near-miss" subgraphs that share some but not all ring properties).

### 16.4 Why this determines everything downstream

- **Loader**: node-seeded vs. edge-seeded vs. subgraph-batched loaders are genuinely different PyG APIs (Lesson 14.4 vs. 14.5 vs. custom subgraph extraction) — you cannot "decide later," since the graph-construction choice (Lesson 15) and the loader are tightly coupled to the target.
- **Decoder**: node-embedding classifier head vs. edge/pair decoder vs. graph-pooling decoder are architecturally distinct components bolted onto the same GNN encoder backbone.
- **Loss**: while the loss *family* (weighted BCE, focal, etc. — Lesson 17) is similar across targets, the *level* at which it's computed (per-transaction, per-account, per-subgraph) changes what a "positive" and "negative" example even means, and therefore how imbalance is measured and corrected.
- **Evaluation**: PR-AUC/F1 must be computed at the same granularity as the label — an account-level PR-AUC and a transaction-level PR-AUC are not comparable numbers and answer different business questions ("which accounts should we investigate" vs. "which transfers should we block").

---

## Lesson 17 — Extreme Imbalance

AML datasets are almost definitionally extremely imbalanced — genuinely laundering-related transactions/accounts/rings are a tiny fraction of all activity (often well under 1%). Training naively (plain BCE, accuracy as the metric) on such data produces a model that trivially predicts "not suspicious" for everything and still scores >99% accuracy while being completely useless.

### 17.1 Weighted BCE

Standard binary cross-entropy weights every example equally:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_i \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$
**Weighted BCE** upweights the minority (positive/suspicious) class by a factor $w$ (often set to roughly the inverse class-frequency ratio, e.g., negatives:positives = 500:1 → $w \approx 500$):
$$\mathcal{L}_{\text{wBCE}} = -\frac{1}{N}\sum_i \left[w\cdot y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$
This makes false negatives on the rare positive class costlier during training, directly counteracting the model's default tendency to ignore the minority class.

### 17.2 Focal loss

**Focal loss** (Lin et al., 2017, originally for object detection's extreme foreground/background imbalance — directly analogous to AML's suspicious/legitimate imbalance) goes further: rather than just reweighting by class frequency, it **downweights easy, already-well-classified examples** (of either class) and focuses training signal on hard, ambiguous ones:
$$\mathcal{L}_{\text{focal}} = -\frac{1}{N}\sum_i \alpha_i (1-\hat{p}_i^{(t)})^\gamma \log(\hat{p}_i^{(t)})$$
where $\hat{p}_i^{(t)}$ is the predicted probability of the *true* class, $\gamma \ge 0$ is the focusing parameter (higher $\gamma$ → more aggressive downweighting of easy examples), and $\alpha_i$ is an optional class-balance weight (combinable with the weighted-BCE idea above). When a transaction is already confidently classified correctly, $(1-\hat p^{(t)})^\gamma \to 0$, so it contributes almost nothing to the gradient — training effort concentrates on the genuinely hard/borderline cases, which in an AML setting are exactly the near-miss suspicious patterns that most need the model's attention.

### 17.3 Sampling strategies

Complementary (not mutually exclusive with the loss reweighting above) approaches operating on the *data* rather than the loss function:
- **Undersampling** the majority (legitimate) class per mini-batch, so each batch has a much less extreme ratio than the true population — cheap, but throws away potentially useful negative examples.
- **Oversampling** the minority (suspicious) class (simple repetition, or synthetic techniques like SMOTE adapted to graph-structured/embedding space) — risk of overfitting to the (few) real positive examples if oversampled too aggressively.
- **Curriculum / hard-negative mining**: deliberately including negatives that are "close" to the decision boundary (e.g., legitimate accounts that share superficial features with flagged ones) rather than random negatives, so the model learns a sharper decision boundary instead of an easy one based on trivially-different negatives.

For graph data specifically, sampling must be done carefully with respect to the loaders in Lesson 14 — e.g., `LinkNeighborLoader`'s `neg_sampling_ratio` controls how many synthetic negative edges are drawn per positive edge per batch, which is a direct lever for both undersampling/oversampling balance and (via `neg_sampling` strategy choice) hard-negative mining.

### 17.4 PR-AUC over ROC-AUC

Under extreme imbalance, **ROC-AUC is misleadingly optimistic**: because it's computed against the (huge) number of true negatives, a model can achieve a very high ROC-AUC while still producing an unusable number of false positives relative to the (tiny) number of true positives — the false-positive *rate* looks small only because the negative class is so large. **PR-AUC (precision-recall AUC)** instead directly tracks the trade-off between precision (of the alerts you'd actually raise) and recall (of the true positives you catch) as the decision threshold varies, without being diluted by the enormous true-negative count — this is why PR-AUC (not ROC-AUC) is the standard primary metric for AML, fraud, and other extreme-imbalance detection tasks.

### 17.5 F1 and threshold optimization

**F1** ($= 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}$) summarizes precision/recall at a *single, chosen* decision threshold — but under imbalance, the default $0.5$ threshold is almost always wrong (the model's raw probabilities are skewed by the class imbalance seen during training, especially with class weighting/focal loss active). **Threshold optimization** means explicitly sweeping thresholds against a held-out validation set (chronologically held out, per Lesson 11.4's leakage warning) and picking the threshold that maximizes F1 — or, more realistically for an operational AML system, picking a threshold that satisfies a *business constraint* (e.g., "investigators can review at most 200 alerts/day" → pick the threshold that yields that alert volume, then report precision/recall at that operating point) rather than optimizing F1 in the abstract.

---

## Lesson 18 — Build the Entire Model in PyTorch Geometric

This lesson is the synthesis lesson where every earlier concept becomes one runnable pipeline. Below is the **scaffold** tying every prior lesson to a concrete PyG component — the actual full build (with your real IBM AML data schema plugged in) is naturally a hands-on coding session rather than something to fully write out in static notes, but here is exactly how each lesson maps into code, so you know precisely what you're building and why at each step:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
from torch_geometric.loader import LinkNeighborLoader

# ---- Lesson 15: graph construction (Option B/C sketch) ----
data = HeteroData()
data['account'].x = account_features          # [n_acc, d_acc]
data['transaction'].x = transaction_features   # [n_txn, d_txn]
data['bank'].x = bank_features                 # [n_bank, d_bank]        (Option C)
data['account', 'initiates', 'transaction'].edge_index = init_ei
data['transaction', 'received_by', 'account'].edge_index = recv_ei
data['account', 'owns_by', 'bank'].edge_index = owns_ei                  # (Option C)
data['transaction'].y = transaction_labels     # Lesson 16: transaction classification target
data['transaction'].time = transaction_timestamps   # Lesson 10/11: for time-based splitting

# ---- Lesson 8/9: heterogeneous, relation-specific GNN layers ----
class HeteroGNN(nn.Module):
    def __init__(self, hidden_dim, out_dim, metadata, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):                       # Lesson 2.6/13: depth = receptive field
            conv = HeteroConv({
                edge_type: GATConv((-1, -1), hidden_dim, add_self_loops=False)  # Lesson 6: attention
                if edge_type[1] in ('initiates', 'received_by')
                else SAGEConv((-1, -1), hidden_dim)                              # Lesson 5: scalable mean agg
                for edge_type in metadata[1]
            }, aggr='sum')                                  # Lesson 9: relation-specific transforms, summed
            self.convs.append(conv)
        self.classifier = Linear(hidden_dim, out_dim)        # Lesson 16: decoder head on transaction node

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = {k: F.relu(v) for k, v in conv(x_dict, edge_index_dict).items()}
        return self.classifier(x_dict['transaction'])         # transaction-level prediction

# ---- Lesson 14: scalable mini-batch loading ----
train_loader = LinkNeighborLoader(
    data,
    num_neighbors=[15, 10],                                    # Lesson 13/14: bounded receptive field
    edge_label_index=(('account', 'initiates', 'transaction'), train_edge_index),
    edge_label=train_labels,
    batch_size=512,
    shuffle=True,
)

model = HeteroGNN(hidden_dim=64, out_dim=1, metadata=data.metadata())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ---- Lesson 17: imbalance-aware loss ----
def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - p_t) ** gamma * ce).mean()

# ---- Training loop ----
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:                                  # Lesson 11.4: batches drawn respecting time-based split
        optimizer.zero_grad()
        out = model(batch.x_dict, batch.edge_index_dict)
        loss = focal_loss(out.squeeze(), batch['transaction'].y[:batch['transaction'].batch_size].float())
        loss.backward()
        optimizer.step()

# ---- Lesson 17: evaluation at the right operating point ----
from sklearn.metrics import precision_recall_curve, auc
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)
```

Every commented line above is a direct callback to a specific earlier lesson — that traceability (which architectural choice solves which problem) is exactly the habit of mind this course is trying to build, so that when you sit down with the real IBM AML dataset, every design decision (graph schema, layer type, loader, loss, split strategy, metric) is a deliberate choice justified by the properties of *that* data and *that* task, rather than a default copied from a tutorial.

---

## Quick-Reference Summary Table

| Lesson | Core question answered |
|---|---|
| 1 | What does it mean to "learn a representation" of a graph, node, edge? |
| 2 | How does one layer of a GNN actually compute a new node representation? |
| 3 | How is that computation expressed efficiently as matrix multiplication? |
| 4 | What exactly is GCN, derived term by term? |
| 5 | How do you make message passing scale and generalize to new nodes? |
| 6 | How do you let the model learn which neighbors matter? |
| 7 | Which of GCN/SAGE/GAT should you reach for, and when? |
| 8–9 | How do you handle graphs with multiple node/edge types? |
| 10–12 | How do you handle graphs that change over time, causally and efficiently? |
| 13–14 | How do you train a GNN when the graph is too big for full-batch training? |
| 15–18 | How do you turn a specific problem (AML) into a concrete graph, target, loss, and PyG pipeline? |

*Sources: W. L. Hamilton, Graph Representation Learning (2020), esp. Ch. 4–5 (encoder-decoder framework, neural message passing, GCN/GraphSAGE/GAT); Y. Ma & J. Tang, Deep Learning on Graphs (2021), esp. chapters on graph convolutional networks, graph attention networks, and heterogeneous/dynamic graph neural networks. Original architecture papers referenced: Kipf & Welling (GCN, 2017), Hamilton/Ying/Leskovec (GraphSAGE, 2017), Veličković et al. (GAT, 2018), Schlichtkrull et al. (R-GCN, 2018), Xu et al. (TGAT, 2020), Rossi et al. (TGN, 2020), Lin et al. (Focal Loss, 2017).*
