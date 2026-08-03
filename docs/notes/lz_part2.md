> **Status:** NOTE &nbsp;·&nbsp; **Updated:** 2026-06-08
>
> Continuation of `lz_molecular_generation.md` — starts at section 4, spectrum-to-DAG generation.

## 4. More Advanced Idea, Spectrum-to-DAG Generation

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
            add branch to DAG              # Case 1 - free, M unchanged
        else:
            queue_edit(M, F_p, protected=atoms_in_matched_fragments)
4.  apply minimal edits (Case 2)
5.  rebuild DAG, score with Branch B → iterate
```

| | Case 1 | Case 2 |
|---|---|---|
| Trigger | F_p reachable from M's atoms; BFS depth-limited | F_p cannot be any atom subset of M |
| Operation | Add DAG branch; M unchanged | Edit M to introduce subgraph with formula F_p |
| Generation needed | No | Yes - scaffold decoration at specific attachment points |
| Matched fragments | Preserved | Protected atoms fixed |

**Noise limitation.** Formula-coincidental matches at depth 3–4 receive IOS reward under
CE. Effective only for genuinely depth-limited fragments; discrimination margin between
correct and wrong candidates matters more than absolute AR.

### 4.2 Native DAG Diffusion

Diffuse directly on the molecular graph; recompute DAG(M) = BFS(M) at each denoising
step; use Branch B as the score function ∇ log p(M | S). Molecule and DAG are the same
object - no rebuild inconsistency.

| Challenge | Description |
|---|---|
| Variable-size DAGs | Fragment count varies widely across molecules |
| Discrete topology | Which bonds break is discrete; standard diffusion is continuous |
| Root recovery | Root node uniquely determines the entire DAG - must be consistent top-down and bottom-up |
| Shared substructures | Same formula fragment can appear under multiple parents |

Related work: Junction Tree VAE (tree of substructures, VAE, no spectrum conditioning);
DiGress (graph diffusion, no DAG, no spectrum); FRIGID (SAFE string diffusion, DAG is
external).