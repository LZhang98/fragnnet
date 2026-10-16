# Formal writing rules

> **Status:** REFERENCE · **Updated:** 2026-10-16
> Style and conventions for formal academic writing, including theses and manuscripts.

## Scope

Apply this guidance when drafting, revising, or reviewing formal academic prose.
It does not govern research plans, implementation plans, task records, progress
updates, or routine project documentation. For those documents, follow
[documentation rules](rules/documentation.md) and the
[session workflow](workflow/session.md).

The thesis paths, build commands, and source examples below describe the thesis
project. Use them only when working in that project and after verifying its
layout; they do not describe the FraGNNet repository.

## Non-negotiable rules

1. Never fabricate citations. If a required citation is unknown, prefer a
   descriptive `% TODO: cite ...` comment that identifies the claim requiring
   support; a bare `\cite{}` placeholder is acceptable when a TODO comment
   would be awkward (e.g., inline mid-sentence). Never invent authors, titles,
   venues, or publication years.
2. Ask when scientific content, the intended argument, or the correct technical
   phrasing cannot be established from the repository or supplied evidence.
3. Do not create thesis figures unless the user explicitly overrides this rule.
   During review-only tasks, report the missing figure without changing the
   file. When the user requests an edit that requires a figure placeholder,
   preserve an existing placeholder or add a blank figure environment with a
   descriptive TODO comment, then ask what the figure should contain.

## Build and project layout

There is no Makefile. Build from the repository root with:

```bash
pdflatex -shell-escape ualberta.tex
bibtex ualberta
pdflatex -shell-escape ualberta.tex
pdflatex -shell-escape ualberta.tex
```

Shell escape is required for nomenclature generation, and three `pdflatex`
passes are required to resolve the table of contents, cross-references, and
bibliography. Because `-shell-escape` permits TeX sources to execute commands,
run this build only for trusted repository content. Inspect newly added TeX
inputs, packages, and command-execution directives before building. If citations
appear as `[0]` after adding BibTeX entries, remove the generated `ualberta.aux`,
`ualberta.bbl`, and `ualberta.blg` files and rebuild. Treat generated files
cautiously and delete them only when rebuilding or when the user asks.

- `00_LaTeX_Files/` contains package, macro, theorem, and formatting setup.
- `01_Prefatory/` contains the front matter.
- `02_Chapters/` contains active thesis chapters.
- `03_References/References.bib` is the only bibliography loaded by default.
- `04_Appendices/` contains appendix material.
- `99_Inclusions/` contains images, tables, code, and included PDFs.
- `other_docs/` contains source and reference material that is not compiled
  into the thesis.

The thesis uses `\insertchapter{name}`, `\insertappendix{name}`,
`\insertprefatory{name}`, and `\insertlatexfile{name}` to resolve files in the
corresponding repository directories. Place chapter images in
`99_Inclusions/Images/`; the configured `\graphicspath` allows the filename to
be used without its full path.

## Academic writing workflow

For thesis drafting, revision, and review, read `docs/academic-writing/README.md`
before working. It defines the evidence, review, editing, and completion process
for this repository. This file remains the canonical source for writing style,
LaTeX conventions, and non-negotiable rules; do not duplicate its voice profile.
Read the target chapter, relevant appendix, and existing `todo/` record, when
present, together with the sources needed for the requested task. Follow existing
authorization and scope; ask only for scientific decisions that the available
evidence cannot resolve. Review tasks remain read-only. Use separate agents only
when the user explicitly requests delegation or parallel agent work.

### Chapters

`ualberta.tex` compiles five chapters from `02_Chapters/` via `\insertchapter{}`:

1. `01_Introduction.tex` — background in metabolomics/MS, problem statement
   (Chemical Dark Matter), related work.
2. `02_NPS_MS.tex` — transfer learning on CFM-ID 4.0 for Novel Psychoactive
   Substances spectrum prediction.
3. `03_FraGNNet.tex` — probabilistic GNN-based fragmentation model for MS/MS
   spectrum prediction. Inputs math macros from
   `other_docs/fragnnet/math_commands.tex`.
4. `04_WithCLM.tex` — chemical language model for reference-free structure
   elucidation; contains two contributions, DarkNPS (NPS chemical space) and
   DeepMet (broader metabolite space).
5. `05_Conclusion.tex` — discussion and conclusion (file not yet present).

### LaTeX configuration files

`00_LaTeX_Files/` files are loaded in order by `ualberta.tex`:

- `includePackages.tex` — all `\usepackage` declarations. Bibliography uses
  `biblatex` with `backend=bibtex`, `style=ieee`, `refsegment=chapter`
  (per-chapter bibliographies via the `chapterbib` class option).
- `includeMacros.tex` — custom tabularx column types (`C` centered auto-width,
  `L` left-justified, `R` right-justified); `\arraystretch=1.5`.
- `includeTheorems.tex` — theorem/definition/lemma environments.
- `listingCodeFormatting.tex` — `listings` package styling.
- `selectDegree.tex` / `selectDepartment.tex` — uncomment the active line to
  change degree or department.

## Thesis Writing Guidance

When drafting or revising thesis prose, preserve the author's established academic voice.

### Source basis and scope

This voice profile is grounded primarily in the thesis and candidacy prose in
`other_docs/`. Use the following passages as examples of how explanations are
developed, rather than as text to copy:

| Source | Relevant writing patterns |
| --- | --- |
| `other_docs/master_thesis/introduction.tex` and `background.tex` | Define the scientific task, motivate its importance, and explain a physical process through its components and stages. |
| `other_docs/master_thesis/appendix.tex` and `conclusion.tex` | Organize related methods by their operation, trace changes between methods, and connect limitations to possible extensions. |
| `other_docs/candidcay/candidcay.md`, Sections 1.2--1.3 and 2.2--2.5 | Explain identification workflows, distinguish what is known from what must be predicted, motivate representations with examples, and explain what each evaluation measures. |
| `other_docs/npsms/Manuscript File for Publication.docx` and `supplement information for publication.docx` | Secondary examples of task-specific evaluation, quantitative comparisons, and interpretation of individual molecule examples. |
| `other_docs/fragnnet/tmlr.tex`, Related Work, Methods, Experiments, and Discussion | Secondary examples of method tradeoffs, operational explanations followed by notation, controlled comparisons, and bounded interpretations. |

The NPS-MS and FraGNNet manuscripts are coauthored; use their shared exposition
as secondary context without attributing every passage to the thesis author.
Treat the papers in `other_docs/clms/` as research context unless the author
identifies passages to use for voice calibration. Exclude journal templates,
bibliographies, quotations, commented-out alternatives, and editorial notes
from the voice profile.

Preserve the explanatory habits of these samples while correcting grammar,
redundancy, and inconsistent terminology. Their historical database counts,
scientific claims, draft placeholders, and promotional wording are not style
rules or automatically verified evidence. The thesis conventions below take
precedence over source-specific formatting. In particular, the candidacy
document describes proposed work; its future tense does not establish that an
experiment was completed.

### Writing style

- Use an explanatory, methodical, and problem-driven structure: establish the scientific problem, explain the limitations of existing methods, introduce the alternative, describe how it works, and evaluate its consequences.
- Explain technical ideas operationally: state what enters a system, what happens, and what the system produces.
- Introduce a method or representation by explaining the problem it addresses. Then describe its components and explain why each operation is needed, using purpose clauses such as `to ...` or `so that ...`.
- Distinguish model training from prediction or retrieval at inference time. State what information is available at each stage and what the model must infer.
- Avoid unnecessarily mathematical phrasing. When mathematical terminology or notation does not add precision, describe the concept directly in operational language.
- Make logical relationships explicit. Use contrasts and caveats to clarify the scope and limitations of claims.
- Prefer cautious academic claims over promotional or absolute language.
- Organize complicated ideas into categories, stages, or enumerated limitations when useful.
- Begin paragraphs with the broader point and progressively introduce technical detail.
- Use moderately long sentences, often around 20--25 words, with shorter definitions and longer explanations when needed. Preserve variation in sentence length; do not pad or split sentences merely to meet a word-count target.
- Let paragraphs develop one explanatory purpose through several connected sentences. The samples often sustain four to six sentences, but paragraph length should follow the argument: split when moving from a mechanism to a different comparison, limitation, or evaluation question.
- Repeat a precise technical noun when doing so is clearer than using an ambiguous pronoun, but avoid unnecessary repetition within the same paragraph.
- Use third person by default. Prefer active first-person plural for thesis contributions and experimental choices: `we introduce`, `we formulate`, `we evaluate`, `we compare`, and `we find`. Use impersonal constructions for generally accepted processes or properties.
- Use present tense for definitions, method behavior, and what a figure or table shows; use past tense for completed experimental procedures and observations. Use future or conditional language for proposed experiments and possible extensions, according to their actual status.
- Define a concept in words before introducing its mathematical notation. After an equation, explain the role or interpretation of its important terms in prose.
- Describe workflows chronologically with signals such as `First`, `Then`, `Following`, and `Finally`. State both the operation and its purpose or output at each stage.
- Introduce abbreviations and directional task names explicitly, then use them consistently. When two directions are easy to confuse, contrast them directly.
- Define a representation through the meaning of its parts. For example, explain what a node, edge, fingerprint bit, or peak denotes before describing how a model processes it.
- Use parenthetical examples and clarifications, including `such as`, `i.e.`, and `e.g.`, when they make an abstract statement concrete. Do not overload a sentence with multiple parenthetical remarks.
- Refer explicitly to figures, tables, sections, and appendices when they carry supporting evidence. Briefly state what the reader should learn from the referenced item.
- In results prose, establish the evaluated task and comparison, report the outcome with its supporting metric, and interpret its consequence. Lead with the outcome when the setup is already clear; introduce the setup first when the task, candidate set, or metric changes. Prefer concrete comparative language such as `outperformed`, `matched`, `higher`, `lower`, `increased`, or `decreased` over vague claims of improvement.
- When discussing a limitation, explain its practical consequence and, when appropriate, follow it with a possible remedy or future direction.
- End major discussions with a compact synthesis of what the evidence establishes, while preserving uncertainty that the experiment cannot resolve.

### Preferred language patterns

Use the following expressions naturally where appropriate; do not insert them mechanically:

- `However, ...`
- `While ..., ...`
- `Note that ...`
- `For example, ...`
- `For instance, ...`
- `More specifically, ...`
- `That is, ...`
- `In addition, ...`
- `In contrast, ...`
- `For comparison, ...`
- `Therefore, ...`
- `To address this limitation, ...`
- `Finally, ...`
- `This approach ...`
- `Our model ...`
- `Given ...`
- `based on ...`
- `can be viewed as ...`
- `can be used to ...`
- `is constrained to ...`
- `does not require ...`
- `in terms of ...`
- `a set of ...`
- `the performance of ...`
- `At a high level, ...`
- `The goal of ... is to ...`
- `The results are summarized in ...`
- `For more details, refer to ...`
- `In summary, ...`
- `In practice, ...`
- `In theory, ...`
- `Compared with ...`
- `Rather than ..., we ...`

Favor precise paired terminology when it clarifies direction or comparison, such as `spectrum-to-structure` and `structure-to-spectrum`, `input` and `output`, `prediction` and `retrieval`, `known` and `unknown`, `reference-based` and `reference-free`, `parent` and `child`, or `observed` and `predicted`.

A characteristic argumentative pattern is:

> While [existing method] can [advantage], it is limited by [problem]. Therefore, [proposed approach] uses [mechanism] to [goal].

Treat this guidance as a voice profile rather than a rigid template. Preserve technical accuracy, grammatical correctness, and variation in sentence structure.

### Citations and explanation

- Integrate citations with the specific method, dataset, or claim they support.
  Introduce an author narratively when explaining a particular contribution or
  how one method extends another; otherwise, keep the method or scientific
  process as the subject and place the citation beside the supported statement.
- In related work, first identify the shared task or representation, then
  explain the mechanism and distinguishing change of each relevant method.
  Use chronology when it explains an extension, rather than listing papers
  without connecting their contributions.
- Define an evaluation metric in terms of what it measures before interpreting
  its values. Explain whether it measures peak agreement, candidate ranking,
  structural similarity, or another property, and state whether higher or lower
  values indicate better performance when this is not already clear.
- Use a concrete example after the general explanation to show how the
  mechanism works or why a metric behaves as reported. State what the example
  illustrates and keep its evidential scope distinct from aggregate results.
- Retain the thesis's citation commands and cross-reference conventions when
  adapting source prose. Do not copy numeric citation text, Zotero document
  links, or manuscript-specific macros as substitutes for resolved references.

### LaTeX and domain conventions

- Introduce key thesis terms with `\textbf{}` when appropriate and use
  `\emph{}` or `\textit{}` for Latin phrases or technical terms on first use.
- Use `\addacronym{KEY}{Full Name}` in `01_Prefatory/Acronyms.tex`, then refer
  to the acronym with `\gls{KEY}` in body text.
- Add nomenclature entries with
  `\nomenclature[<group>]{<symbol>}{<description>}`, where `A`, `B`, `C`, and
  `D` denote constants, Latin symbols, Greek symbols, and subscripts.
- Write cross-references in full, such as `Chapter~2`, `Section~2.4`, and
  `Figure~3.1`.
- Capitalize *Chemical Dark Matter* consistently. Use `MS/MS spectra`, not
  informal abbreviations such as `mass spec` or an ambiguous unqualified
  `spectra`.
- Use `reference-free` as the thesis-defining adjective and contrast it
  explicitly with `reference-based` where relevant.
- Introduce directional method categories explicitly and use a consistent
  notation, such as `structure-to-spectrum` and `spectrum-to-structure`. Retain
  arrow notation only where the surrounding thesis text already uses it.
- Keep model names in plain text after their first introduction unless an
  established LaTeX macro specifies otherwise.
- Prefer `improve` over `enhance`. Avoid em dashes; use commas, semicolons,
  colons, or parentheses.

### Thesis-specific argumentative patterns

- Anchor abstract MS/MS concepts with concrete molecules or measured examples
  when the evidence is available, such as caffeine, bisoprolol, tyrosine,
  ocfentanil, 3-Cl-PCP, methylaminomethanol, or adiphenine.
- Use exact quantitative database comparisons only when they are supported by
  a citation or verifiable repository source. Preserve the database version or
  date when a count describes a historical snapshot.
- Use `Recall that ...` or `Recall our definition of ...` to reconnect later
  arguments to previously defined concepts without repeating the definition.
- For several technical obstacles, use a numbered decomposition and connect
  each obstacle to its practical mechanism or consequence.
- Frame chapter introductions as problem, existing approaches, limitation, and
  contribution. State contributions in prose rather than a bulleted list.
- In results prose, state the controlled comparison and its quantitative
  consequence in the same paragraph. The comparison may open the paragraph or
  provide its closing synthesis, but do not repeat the same result merely to
  occupy both positions. In discussions, make causal isolation explicit, for
  example by stating which comparison isolates one mechanism from another.
- When linking spectrum prediction to compound identification, explain the
  intervening retrieval workflow and report its evaluation separately. Keep
  exact identification, structural similarity, and fragment interpretation
  distinct when discussing what an experiment establishes.
- Explain a design choice by comparing the relevant alternatives and their
  practical tradeoff, then state why the selected approach suits the task.
  Identify additional assumptions or information available to a benchmark
  variant before interpreting its performance.
- End a chapter with a forward bridge that identifies the next chapter's
  mechanism, not only its topic.

### Argument and paragraph patterns

Use topic sentences that make the paragraph's function clear. Common paragraph structures in the author's writing include:

1. Define a problem or concept, provide a concrete example, and explain why it matters.
2. Describe an existing method, identify its benefit, and then state its limitation.
3. Introduce the proposed method, explain its stages in order, and conclude with the capability enabled by those stages.
4. Point the reader to a result, report the principal comparison, and explain the likely reason or implication.
5. State a limitation, identify the evidence for it, and propose a technically plausible remedy.
6. Define an evaluation task, explain the metric and comparison, report the result, and use an individual example to clarify a success or failure when useful.

For comparisons involving several methods, keep a consistent comparison axis. A useful pattern is:

> [Method A] benefits from [advantage]; however, it is limited by [drawback]. In contrast, [Method B] provides [different advantage], but [remaining drawback]. Our approach combines [relevant strengths] by [mechanism].

### Claim strength

- Use `show` or `demonstrate` when the stated conclusion is directly supported by the reported experiment.
- Use `suggest`, `indicate`, `may`, `could`, or `potential` for interpretations, generalization beyond the evaluated data, and future applications.
- Use qualifiers such as `typically`, `often`, `generally`, `approximately`, and `to our knowledge` when their scope is meaningful and defensible.
- Avoid hiding a concrete result behind excessive hedging. Give the measured result first, then qualify its interpretation.
- Distinguish what the model establishes from what remains unknown; formulations such as `Although we cannot establish ..., our experiments indicate ...` fit the author's preferred style.
- Separate an observed result from a proposed explanation. State competing explanations when the experiment cannot distinguish them, and tie a proposed remedy to the specific limitation it would address.

### Patterns to avoid

Unless required by the genre, a quotation, or the underlying evidence, do not:

- Ask rhetorical questions. State the motivating problem or uncertainty directly.
- Address the reader as `you` or give conversational commands such as `consider` or `imagine`.
- Use first-person singular (`I`, `me`, or `my`). Use `we` and `our` for thesis contributions and experimental choices, including in single-author thesis prose; use third person or impersonal constructions elsewhere.
- Use contractions such as `don't`, `isn't`, `can't`, or `we've`.
- Use exclamation marks, jokes, personal anecdotes, or casual conversational asides.
- Use emphatic certainty markers such as `obviously`, `clearly`, `certainly`, `definitely`, or `undoubtedly` as substitutes for evidence.
- Claim that an empirical result `proves` a general conclusion. Prefer `shows` or `demonstrates` for direct evidence and `suggests` or `indicates` for interpretations.
- Use emotional or promotional modifiers such as `amazing`, `remarkable`, `remarkably`, `surprising`, `surprisingly`, or `strikingly` unless the comparison makes the description precise and defensible.
- Prefer ornate transitions such as `furthermore`, `nevertheless`, or `nonetheless` when a characteristic transition such as `however`, `while`, `in addition`, or `in contrast` is clearer.
- Begin a concluding section with the stock phrase `In conclusion`. Begin with the principal finding directly or use `In summary` when an explicit synthesis marker is useful.
- Use extended metaphors, decorative historical narratives, or analogies that do not materially clarify a technical idea.
- Use sentence fragments or a sequence of very short, punchy sentences for rhetorical effect. Short sentences remain appropriate for precise definitions and necessary emphasis.
- Dismiss competing methods or characterize them as simply wrong. Describe their relevant advantages, assumptions, and limitations using a consistent comparison axis.
- Leave the basis of a comparison implicit. Identify whether methods differ in accuracy, efficiency, coverage, interpretability, generalization, or another relevant property.
- End a section with an unsupported broad claim. Tie the synthesis to evidence, acknowledge unresolved uncertainty, or identify a concrete direction for future work.
- Reproduce draft artifacts from the source samples, including unfinished sentences, repeated words, spelling errors, unresolved editorial questions, or unsupported superlatives.
