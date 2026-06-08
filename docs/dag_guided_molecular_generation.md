# DAG-Guided Molecular Generation from Mass Spectra

Reference: FRIGID (Bohde et al., 2026) — https://arxiv.org/abs/2604.16648

---

## 1. FRIGID — Mechanics

Diffusion LM over SAFE sequences conditioned on spectrum. At inference, uses ICEBERG as
a forward oracle to iteratively remask tokens:

- Generate M → run ICEBERG → match simulated peaks to observed (±20 ppm)
- Matched fragment atoms: `+1`; hallucinated fragment atoms: `−γ`
- Aggregate to SAFE token level via **min** over constituent atoms → remask → re-denoise

---

## 2. Problems with FRIGID

**1. `min` aggregation penalises correct deep matches.**
A matched child F2 whose atoms are a subset of a hallucinated parent F1 gets
`min(+1, −γ) = −γ`. Correct evidence at depth 2 is erased by the hallucination at depth 1.

```mermaid
flowchart TD
    P["Precursor"]
    F1["F1  m/z 300\nhallucinated → −γ"]
    F2["F2  m/z 150\nmatched (atoms ⊂ F1) → min = −γ"]
    P --> F1 --> F2
    style F1 fill:#ffdddd,stroke:#cc4444
    style F2 fill:#ffeecc,stroke:#cc8844
```

**2. Remasking is DAG-blind.**
Top-down order is required: find the shallowest hallucination, protect matched children,
then remask the parent. ICEBERG's flat fragment list makes this impossible.

**3. No proper posterior.**
Iterative remasking is not an approximation to any well-defined distribution;
P(M | S) ∝ p(S | M) × p(M) is never computed.

---



## 3. Spectrum-to-Molecule Posterior Architecture

**Goal:** one MS/MS spectrum → calibrated distribution over generated molecules.

- Not: spectrum → fingerprint → one molecule
- Not: spectrum → one molecule
- Output: `M₁: 0.41   M₂: 0.19   M₃: 0.08   …`

```
P(M | S) ∝ p(S | M) × p(M)
P(Mₖ | S) ≈ softmax_k[ f(S, Mₖ) + log p(Mₖ) ]
```

```mermaid
flowchart TD
    S(["Observed spectrum S"])
    E["Spectrum encoder"]
    zS(["zS"])

    subgraph A ["Branch A — candidate generator"]
        GA["zS + noise / masked tokens"]
        MA(["M₁  M₂  …  Mₖ"])
        GA --> MA
    end

    subgraph B ["Branch B — scorer  f(S, M)"]
        SC["fragment-aware\nscoring"]
    end

    S --> E --> zS --> GA
    MA -->|each candidate| SC
    S --> SC
    SC --> SM["softmax over k"]
    SM --> PM(["P(M | S)"])
```

*Figure 1. Branch A generates candidates; Branch B scores each; softmax yields the posterior.*

### 3.1 Branch A — Molecule Generator

Branch A learns P(M | S): one spectrum → 100–1000 candidate molecules.

| | |
|---|---|
| **Architecture** | SAFE / SELFIES / graph diffusion / flow matching |
| **Pretraining** | Unconditional molecule LM on large databases |
| **Fine-tuning** | Conditional generation on paired (spectrum, molecule) data |
| **Inference** | Sample candidates; filter invalids and duplicates |

### 3.2 Branch A Options

| Option | When to use |
|---|---|
| **Database retrieval** | Known chemical space; Branch B alone gives the posterior |
| **Frozen generator** (FRIGID-style, trained independently) | De novo, inference-time composition |
| **Joint ELBO** (generator maximises E[log p(S\|M)] − KL) | Maximum performance; needs reparameterisable generation |

### 3.3 Branch B — Fragment-Aware Scorer

`f(S, M)` approximates `log p(S | M)` on a **comparable scale** across all candidates.
Inputs: observed peak m/z and intensities; candidate fragment masses and formulae;
matched vs. unmatched peaks; hallucinated predicted fragments.

Branch A quality is a hard floor: if proposal `q(M | S)` is poor, Branch B selects the
least-wrong candidate, not the correct one. The unbiased posterior requires:

```
P(Mᵢ | S) ∝ p(S | Mᵢ) × p(Mᵢ) / q(Mᵢ | S)
```

**GNN speedup.** If Branch B uses a GNN over the fragment DAG and conditions on
CE/adduct/inst at the MLP level, the GNN is context-independent → fragment embeddings
`{h_f}` can be precomputed offline; online cost = one MLP pass + sparse peak matching.

```mermaid
flowchart LR
    subgraph OFF ["Offline — once per molecule"]
        M0["M"] --> B0["BFS"] --> G0["GNN"] --> H0(["stored {h_f}"])
    end

    subgraph ON ["Online — query time  (<1 ms)"]
        H1(["cached {h_f}"])
        C1(["CE · adduct · inst"])
        ML1["MLP + FiLM"]
        P1(["predicted spectrum"])
        SC1["f(S, M)"]
        S1(["observed S"])
        H1 --> ML1
        C1 --> ML1
        ML1 --> P1 --> SC1
        S1 --> SC1
    end

    H0 -. "lookup" .-> H1
```

*Figure 2. Precomputed GNN embeddings make softmax over database-scale candidate sets tractable.*

### 3.4 Training

| Stage | Objective |
|---|---|
| **1 — Generator pretraining** | Unconditional molecule LM on SMILES / SELFIES / SAFE |
| **2 — Spectrum-conditioned generation** | Branch A conditional generation loss (spectrum → true molecule) |
| **3 — Scorer training** | InfoNCE with hard negatives: same-formula, same-mass, same-class, random decoys |
| **4 — Joint fine-tuning** | L_generation + λ₁ L_scoring + λ₂ L_fragment_consistency + λ₃ L_calibration |


---

## 4. Spectrum-to-DAG Generation

The observed spectrum is a partially-observed DAG: peak m/z values are node labels;
which molecular subgraphs produce those masses is the inference target. The precursor is
the DAG root.

### 4.1 DAG Branch Regeneration

```
1.  DAG = BFS(M)
2.  matched, unmatched = match_peaks(DAG, S)
3.  for peak p in unmatched (high intensity first):
        F_p = formula_from_mass(p.mz, precursor_formula)
        if subgraph_search(M, F_p):
            add branch to DAG              # Case 1 — free, M unchanged
        else:
            queue_edit(M, F_p, protected=atoms_in_matched_fragments)
4.  apply minimal edits (Case 2)
5.  rebuild DAG, score with Branch B → iterate
```

| | Case 1 | Case 2 |
|---|---|---|
| Trigger | F_p reachable from M's atoms; BFS depth-limited | F_p cannot be any atom subset of M |
| Operation | Add DAG branch; M unchanged | Edit M to introduce subgraph with formula F_p |
| Generation needed | No | Yes — scaffold decoration at specific attachment points |
| Matched fragments | Preserved | Protected atoms fixed |

**Noise limitation.** Formula-coincidental matches at depth 3–4 receive IOS reward under
CE. Effective only for genuinely depth-limited fragments; discrimination margin between
correct and wrong candidates matters more than absolute AR.

### 4.2 Native DAG Diffusion

Diffuse directly on the molecular graph; recompute DAG(M) = BFS(M) at each denoising
step; use Branch B as the score function ∇ log p(M | S). Molecule and DAG are the same
object — no rebuild inconsistency.

| Challenge | Description |
|---|---|
| Variable-size DAGs | Fragment count varies widely across molecules |
| Discrete topology | Which bonds break is discrete; standard diffusion is continuous |
| Root recovery | Root node uniquely determines the entire DAG — must be consistent top-down and bottom-up |
| Shared substructures | Same formula fragment can appear under multiple parents |

Related work: Junction Tree VAE (tree of substructures, VAE, no spectrum conditioning);
DiGress (graph diffusion, no DAG, no spectrum); FRIGID (SAFE string diffusion, DAG is
external).

