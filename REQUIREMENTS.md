# REQUIREMENTS.md — SAR-Assisted Land-Cover Classification Case Study

Source: `Case study - SAR-Assisted.pdf` (Blurgs AI, 3 pages). All quotes below are verbatim from
that document. Page numbers refer to the PDF's page order (p.1 = "Instructions and Expectation",
p.2 = "Case study: SAR-Assisted Understanding" + "Dataset" + "Task" + "Minimum baseline",
p.3 = "Contribution" + "Evaluation").

Throughout, items are tagged:
- **[STATED]** — explicit requirement/constraint in the PDF.
- **[SUGGESTED]** — a reasonable implementation choice inferred by me, not mandated by the PDF.
  These are flagged so they can be revisited/changed freely; they are not commitments.

---

## 1. Exact research problem

**[STATED]** (p.2, "Case study" + "Task"):

> "Your goal is to investigate whether SAR provides useful information when optical observations
> are partially unavailable."

> "Build a model that predicts land-cover classes under three conditions: A. Optical only ...
> B. Degraded optical ... C. Degraded optical + SAR ... Compare performance between B and C."

So the research question is narrowly scoped: **does adding Sentinel-1 SAR to a degraded
(partially cloud-masked) Sentinel-2 optical input improve land-cover classification relative to
using the degraded optical image alone?** It is a multi-class (or multi-label, depending on the
BigEarthNet label scheme chosen) land-cover classification task, not segmentation, detection, or
change detection. The primary comparison the case asks for is **B vs. C**; A is present mainly to
establish the achievable ceiling with clean optical data and to make the degradation's cost
legible.

## 2. Motivation for using SAR

**[STATED]** (p.2):

> "Satellite optical imagery is intuitive to interpret but frequently obscured by clouds. SAR
> imagery can observe the surface under many conditions where optical imagery is unavailable."

The stated motivation is purely about **cloud/atmospheric robustness**: SAR (Sentinel-1, C-band)
penetrates cloud cover and is not dependent on solar illumination, so it can supply information
when optical (Sentinel-2) observations are missing or obscured. No other motivation (e.g.
texture/structure cues, soil moisture, all-weather revisit cadence) is mentioned in the PDF —
those would be **[SUGGESTED]** framing I could add in the README, but they are not requirements.

## 3. Dataset requirements

**[STATED]** (p.2, "Dataset"):

> "Use a manageable subset of paired Sentinel-1 and Sentinel-2 imagery from BigEarthNet. You do
> not need to use the complete dataset. Select several land-cover classes and construct a subset
> that can comfortably be processed on your available hardware."

Explicit constraints:
- Source dataset: **BigEarthNet** (the paired S1/S2 variant, i.e. BigEarthNet-S1/S2 / BigEarthNet-MM).
- **Subsetting is expected and encouraged** — full dataset is explicitly not required.
- Must select **"several" land-cover classes**, not necessarily the full label taxonomy.
- Subset size must be driven by what "can comfortably be processed on your available hardware"
  (no fixed number of samples/classes is given by the PDF).

No requirement is stated about: which BigEarthNet label version (original 43-class vs. the
19-class simplified nomenclature), single-label vs. multi-label framing, train/val/test split
policy, or geographic/seasonal balancing. **[SUGGESTED]** — these are implementation decisions
left open, to be decided (and justified) later, not now.

## 4. The three required experimental conditions

**[STATED]** (p.2, "Task"), quoted in full:

> "A. Optical only: Use Sentinel-2 imagery.
> B. Degraded optical: Artificially mask portions of Sentinel-2 imagery to simulate
> missing/cloud-obscured observations.
> C. Degraded optical + SAR: Use the degraded Sentinel-2 image together with the corresponding
> Sentinel-1 observation.
> Compare performance between B and C."

Notes on what is and isn't specified:
- The masking mechanism for B ("artificially mask portions") is **not specified** — shape,
  intensity, or realism of the simulated cloud mask is left to the implementer. **[SUGGESTED]**
  choices (e.g. random rectangular occlusion vs. synthetic cloud-shaped masks, per-band vs.
  spatial masking) must be documented as design decisions, not treated as given.
  is required (see §8): "different levels of simulated cloud coverage" (p.3), so the masking
  procedure must be parameterizable by a coverage level.
- Condition C must use SAR "together with" the *same degraded* optical image from B, not the
  original clean image — i.e., C's optical branch input should match B's, with SAR added on top.
  This is the controlled-comparison design the B-vs-C ask depends on.
- Condition A (optical only, clean) is required as a condition but the PDF does not explicitly
  ask to "compare A" against anything — it exists to frame the ceiling/reference point.

## 5. Minimum baseline

**[STATED]** (p.2, "Minimum baseline"):

> "Implement a simple CNN or pretrained image encoder using optical imagery. Then evaluate the
> same model after artificially masking part of the optical image."

This defines the floor of acceptable effort for conditions A and B:
- A single model (CNN trained from scratch **or** a pretrained image encoder used as a backbone)
  operating on Sentinel-2 only.
- The **same trained model** re-evaluated on masked/degraded inputs (i.e., the baseline does not
  require retraining a separate model for degraded inputs unless the chosen approach calls for it
  — the PDF's minimum is: train once on clean optical, then test under degradation).
- Explicitly not required to be state-of-the-art or production quality (p.1: "We are not looking
  for a production system or a state-of-the-art model").

This is a **floor**, not the target: the case separately requires an actual SAR-assisted
contribution (§6) built beyond this baseline.

## 6. What constitutes the SAR-assisted contribution

**[STATED]** (p.3, "Contribution"), quoted in full:

> "Develop one SAR-assisted approach.
> Examples include:
> ● concatenating SAR and optical channels;
> ● separate SAR and optical encoders followed by feature fusion;
> ● mapping SAR features into an optical feature space;
> ● using SAR to predict information missing from the optical representation.
> The architecture does not need to be sophisticated."

Requirements:
- **Exactly one** SAR-assisted architecture needs to be developed (not a sweep of several fusion
  strategies) — "Develop one SAR-assisted approach."
  presented as illustrative options ("Examples include"), not a checklist to satisfy — only one
  needs to be chosen and implemented.
- Explicit permission to keep it simple: "does not need to be sophisticated." Architectural
  novelty/sophistication is not an evaluation axis per p.1's stated criteria.
- Which of the four listed patterns (or another reasonable one) to use is **[SUGGESTED]** /
  open — the PDF does not mandate a specific fusion strategy.

## 7. Required evaluation metrics

**[STATED]** (p.3, "Evaluation"):

> "Report an appropriate classification metric such as accuracy, macro F1, and others."

- At minimum, an "appropriate classification metric" — accuracy and macro F1 are given as named
  examples, "and others" leaves room (and mild expectation) for more than one metric.
- No specific metric is mandated as required beyond "appropriate"; the choice should be justified
  given the label scheme (e.g. macro-F1 / per-class F1 matter more than accuracy under class
  imbalance or multi-label framing — this justification is **[SUGGESTED]** analysis, not quoted
  text).
- No mention of calibration metrics (ECE, Brier score, AUROC) as *required* — see §10 on
  uncertainty, which is a separate, more loosely specified expectation.

## 8. Required comparisons and ablations

**[STATED]** (p.3, "Evaluation"):

> "Break the results down by: optical only, cloudy/degraded optical, cloudy/degraded optical +
> SAR, different levels of simulated cloud coverage."

This is an explicit, itemized requirement — the evaluation must report results sliced by:
1. Condition A (optical only)
2. Condition B (degraded optical)
3. Condition C (degraded optical + SAR)
4. **Multiple levels of simulated cloud/mask coverage** (e.g. a sweep such as 0%, 25%, 50%, 75%
   masked area — exact levels are **[SUGGESTED]**, not specified in the PDF), applied presumably
   to both B and C so the B-vs-C gap can be shown *as a function of degradation severity*.

Combined with §1/§4, the core required comparison is **B vs. C across a range of degradation
levels**, with A reported as reference. No other ablations (e.g. per-land-cover-class breakdown,
sensor-only-SAR condition, alternative fusion strategies) are explicitly required, though a
per-class breakdown is a natural **[SUGGESTED]** extension given "select several land-cover
classes" (§3) and the emphasis on scientific validity (p.1).

## 9. Expected visualizations

**[STATED]** (p.1, "A good submission normally contains"):

> "5. Visualisation of representative successes and failures."

This is the only explicit visualization requirement. It calls for qualitative examples — cases
where the SAR-assisted model succeeds (e.g., correctly classifies under heavy degradation where
optical-only fails) and cases where it fails — not a specific plot type. Quantitative plots (e.g.
accuracy/F1 vs. cloud-coverage-level curves per condition) are a natural **[SUGGESTED]** way to
satisfy §8's breakdown requirement visually, but are not separately named as "required
visualizations" in the text — they fall under "Quantitative evaluation" (item 4, same list) rather
than item 5.

## 10. Requirements around uncertainty and failure cases

**[STATED]** (p.1, "We care about"):

> "how you reason about uncertainty and failure cases"

This appears only as one of six evaluation criteria on page 1 — it is **not accompanied by any
methodological specification** (no mention of confidence calibration, predictive uncertainty
estimation methods, ensembling, or specific failure-mode taxonomies). Combined with item 5 above
("visualisation of representative successes and failures") and item 6 ("a short discussion of
what you would try next"), the literal requirement is: identify and discuss where/why the model
is uncertain or wrong (e.g., which classes confuse each other, whether heavier degradation
correlates with lower confidence, whether SAR helps or hurts specific failure modes), not to
implement a particular uncertainty-quantification technique. Any specific technique used to
surface this (e.g. softmax entropy, per-class confusion matrices) is **[SUGGESTED]**.

## 11. Expected submission artifacts

**[STATED]** (p.1, "A good submission normally contains"), quoted in full:

> "1. A reproducible notebook or small codebase.
> 2. A README describing the problem understanding and approach.
> 3. At least one simple baseline and considered approach.
> 4. Quantitative evaluation.
> 5. Visualisation of representative successes and failures.
> 6. A short discussion of what you would try next."

All six are explicitly listed as what a "good submission normally contains" — phrased as a norm
("normally contains") rather than an absolute checklist, but should be treated as the de facto
deliverable list. Item 3 restates the two-tier structure already required by §5/§6 (a simple
baseline, plus one considered/SAR-assisted approach).

## 12. Constraints and things explicitly NOT to spend substantial time on

**[STATED]**:
- (p.1) "This is a 24-48 hour take-home exercise. We are not looking for a production system or a
  state-of-the-art model."
- (p.1) "Do not spend substantial time tuning hyperparameters."
- (p.2) "You do not need to use the complete dataset [BigEarthNet]."
- (p.3) "The architecture does not need to be sophisticated."

Taken together, these four statements set an explicit ceiling on engineering effort: no
production hardening, no SOTA-chasing, no hyperparameter sweeps, no full-dataset-scale training,
no architectural sophistication. Effort should instead go toward the six criteria on p.1
(problem formulation, data inspection/cleaning, sensible baselines, valid evaluation, uncertainty/
failure reasoning, clarity) — this is the implicit trade-off the case is testing.

## 13. Requirements around documenting external libraries, pretrained models, datasets, or code

**[STATED]** (p.1):

> "You may use open-source libraries and pretrained models. Please document any external models,
> data, or code you use."

This is an explicit permission + an explicit documentation obligation: any pretrained encoder
(e.g. an ImageNet or remote-sensing-pretrained backbone), open-source library, or external
code/dataset used must be named and documented (e.g. in the README) — not merely used silently.
No specific format for this documentation is mandated (e.g. no requirement for a formal
citations/bibliography section) — **[SUGGESTED]**: a "External resources used" section in the
README listing library/model/dataset name, version/source, and purpose would satisfy this
plainly.

## 14. Hardware/compute considerations explicitly mentioned

**[STATED]** (p.2): "construct a subset that can comfortably be processed on your available
hardware."

This is the only hardware-related statement in the document. It:
- Confirms subsetting BigEarthNet is expected specifically so the exercise fits whatever hardware
  the candidate has (no GPU/CPU/RAM spec is given, no cloud credits or compute budget mentioned).
- Implicitly reinforces §12 — this is another lever (dataset size) meant to keep the exercise from
  becoming a large-scale training effort.

No other hardware, runtime, or budget constraint appears anywhere in the PDF.

---

## Explicit non-goals recap (for quick reference)

The following are **[STATED]** as *not* required or *not* the point of the exercise:
- Not a production system.
- Not a state-of-the-art model.
- Not the complete BigEarthNet dataset.
- Not substantial hyperparameter tuning.
- Not a sophisticated fusion architecture.
- Not multiple SAR-assisted approaches — one is sufficient.

## Open questions left to a later (design) stage — NOT decided here

Per the task instructions for this stage, the following remain intentionally undecided and are
out of scope for this document: which BigEarthNet label taxonomy/split to use, exact subset size
and class selection, exact cloud-masking algorithm and coverage levels, choice of backbone/
pretrained encoder, which SAR-fusion pattern to implement, and any specific uncertainty-
quantification technique. No datasets have been downloaded, no models trained, and no
architecture chosen at this stage.
