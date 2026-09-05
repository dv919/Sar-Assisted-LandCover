# DATA_ACCESS.md — BigEarthNet v2.0 Access Strategy

Status: **Phase 2 complete.** The two official metadata parquet files (~4.3 MB total) have been
downloaded and inspected in full, and a small (~62 MB total, two range-limited requests) archive-
ordering probe has been run against the front of both `.tar.zst` archives. **No dataset imagery
has been bulk-downloaded, no classes have been finalized/committed, and nothing has been
trained.** Everything below §0–§8 is the original pre-download research pass; **see the "Phase 2
— Empirical Findings" section near the end for what was actually verified by inspection**, which
supersedes the `[UNVERIFIED — needs runtime check]` tags below wherever it resolves them.

Original framing retained below: this document answers the eight questions asked, then makes one
recommendation for approval.

Confidence tagging used throughout:
- **[VERIFIED]** — confirmed directly from an official primary source (bigearth.net, the Zenodo
  record page, or the reBEN paper/authors' own tooling docs).
- **[REPORTED, secondary]** — stated consistently across secondary/community sources (ConfigILM
  docs, GitHub pipeline repo, ecosystem write-ups) but not something I pulled from the primary
  PDF/paper myself; treat as reliable but re-confirm against the official dataset description PDF
  before writing code against it.
- **[UNVERIFIED — needs runtime check]** — a claim that matters for the download plan but can only
  be confirmed by actually probing the files (which I have not done, per instructions to stop
  before downloading).

---

## 0. Official version and source

- **Dataset**: BigEarthNet v2.0, also referred to as **reBEN** ("refined BigEarthNet") in the
  2024 paper by Clasen, Hackel, Burgert, Sumbul, Demir & Markl (TU Berlin, RSiM/DIMA groups).
  **[VERIFIED]**
- **Official project site**: https://bigearth.net/ **[VERIFIED]**
- **Official data host**: Zenodo record **10891137**, version **2.0.0**, published 2024-07-04:
  https://zenodo.org/records/10891137 **[VERIFIED]**
- **License**: Community Data License Agreement – Permissive, Version 1.0 (CDLA-Permissive-1.0).
  **[REPORTED, secondary]**
- v1.0 is a different, older Zenodo record (Sentinel-2-only originally, later paired with a
  separately-published BigEarthNet-S1 v1.0); we are **not** using it — v2.0 is the current
  official version, uses an improved atmospheric correction (sen2cor v2.11) and a refined,
  lower-noise CORINE Land Cover 2018 label set, and is the version the case study asks for
  implicitly by naming "BigEarthNet" without qualification while the field's current default is
  v2.0. No compelling reason to use v1 was found.

## 1. How S1 and S2 data are organized

- **[VERIFIED, Zenodo file listing]** The record hosts these files:

  | File | Size |
  |---|---|
  | `BigEarthNet-S1.tar.zst` | ~54.4 GB |
  | `BigEarthNet-S2.tar.zst` | ~63.3 GB |
  | `Reference_Maps.tar.zst` | ~282 MB |
  | `metadata.parquet` | ~3.6 MB |
  | `metadata_for_patches_with_snow_cloud_or_shadow.parquet` | ~710 KB |

  **Total: ~118 GB** for the two imagery archives combined; metadata is negligible (~4.3 MB).

- **BigEarthNet-S2**: derived from **115 Sentinel-2 L2A tiles** (atmospherically corrected with
  sen2cor v2.11), acquired June 2017–May 2018, over 10 European countries (Austria, Belgium,
  Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, Switzerland).
  **[VERIFIED/REPORTED]**
- **BigEarthNet-S1**: derived from **312 Sentinel-1 GRD scenes** temporally close to the S2
  tiles, preprocessed into patches matching the S2 patch grid — **one S1 patch per S2 patch**.
  **[VERIFIED/REPORTED]**
- Each archive, once extracted, is expected to contain one folder per patch holding per-band
  GeoTIFFs (this per-patch-folder-of-GeoTIFFs layout is how BigEarthNet v1 was structured and
  the v2.0 docs describe the same general packaging); the **exact internal directory tree, file
  naming convention, and byte-order of entries inside the `.tar.zst` are not confirmed by any
  source I read** — the only way to confirm this is to open the archive (or at minimum its tar
  header stream) directly. **[UNVERIFIED — needs runtime check]**. This matters a lot for the
  download strategy (§8).

## 2. How patches are paired

- Patches are indexed **"in alphabetical order based on their S2-name"** according to the
  ConfigILM library's BigEarthNet v2.0 documentation, i.e., the canonical key for a pair is the
  Sentinel-2 patch name, and the corresponding Sentinel-1 patch name is a metadata field attached
  to it (commonly called `s1_name` in adjacent tooling/metadata mapping files).
  **[REPORTED, secondary]**
- The TU Berlin pipeline repo (`rsim-tu-berlin/bigearthnet-pipeline`, the tool used to *build*
  the dataset) generates a `patch_id_s2v1_mapping.csv`-style set of per-patch-id mapping files
  (label mapping, split mapping, country mapping) keyed by patch id — confirming pairing is
  **id-based via a shared patch identifier**, not filename pattern-matching we'd have to invent
  ourselves. **[REPORTED, secondary]**
- Practical implication: once `metadata.parquet` is loaded, each row should give us both the S2
  patch id and its paired S1 patch id (or enough fields to construct both paths) — pairing is a
  metadata lookup, not something to reverse-engineer from imagery. Exact column name to confirm
  on inspection.

## 3. What metadata files are available

- **`metadata.parquet`** (~3.6 MB): metadata for all patches **except** those flagged as covered
  by seasonal snow, cloud, or cloud shadow. **[VERIFIED, Zenodo record]**
- **`metadata_for_patches_with_snow_cloud_or_shadow.parquet`** (~710 KB): metadata for the
  complementary set of patches that *are* affected by snow/cloud/shadow. **[VERIFIED]**
  - These two together should cover all 549,488 pairs; combining them (and keeping a
    `has_snow_cloud_shadow` boolean derived from which file a row came from) is expected to be
    part of standard loading — ConfigILM's datamodule literally requires both files and merges
    them at init. **[REPORTED, secondary]**
- A **dataset description PDF** is linked from bigearth.net's downloads section, described as
  documenting directory structure, naming conventions, and metadata — this is the authoritative
  reference for exact column schema and folder layout and **should be read before writing any
  loading code**, but I have not been able to open/parse it in this pass (only the HTML page
  around it was fetched). **[UNVERIFIED — recommend fetching directly next]**
- Exact column names of `metadata.parquet` (e.g. `patch_id`, `s1_name`, `split`, `labels`,
  `country`, `contains_seasonal_snow`, `contains_cloud_or_shadow`) are **consistent across
  multiple secondary sources' *descriptions*** of what the file contains, but I could not fetch
  a literal schema dump — treat the exact field names as **[UNVERIFIED — needs runtime check]**
  until we load the parquet file's schema directly (trivial once the 3.6 MB file is on disk: this
  alone is a safe, sub-5MB download that respects "don't download 100+GB").

## 4. How official train/validation/test splits are represented

- **[REPORTED, secondary, consistent]** The split is represented as a column (commonly `split`)
  inside `metadata.parquet` with values `train` / `validation` (or `val`) / `test`, not as
  separate archives — i.e., all patches ship in the same imagery archives regardless of split;
  the split is a metadata filter, not a folder/file split.
- The reBEN paper's stated contribution is specifically **"a new geographical-based split
  assignment algorithm that significantly reduces the spatial correlation among the train,
  validation, and test sets"** — i.e., unlike some naive random splits, v2.0's official split is
  explicitly designed to minimize spatial leakage between train/val/test (patches from the same
  tile/neighborhood should not straddle splits in ways that let a model memorize location).
  **[VERIFIED, from paper abstract]**
- Practical implication: **we should use the official `split` column as-is** rather than
  re-splitting ourselves, specifically because re-splitting could reintroduce the spatial leakage
  the authors deliberately engineered against — this is a scientific-validity point directly
  relevant to the case's "whether your evaluation is scientifically valid" criterion.
- Exact split proportions/counts are documented in a "class distribution document" mentioned
  alongside the downloads (train/val/test quantities per land-cover class) — not yet fetched.
  **[UNVERIFIED — needs runtime check / follow-up fetch]**

## 5. Filtering a subset using metadata before downloading imagery

**Yes — this is explicitly supported by the file layout**, and is the intended workflow:
- `metadata.parquet` + the snow/cloud parquet together are ~4.3 MB and contain, per patch: id(s),
  split, label(s), country, and cloud/snow flags (per §3). This is **small enough to download in
  full at zero real cost** and inspect/filter entirely offline before touching either imagery
  archive.
- This lets us pick patch ids for a subset (by class, split, country, cloud/snow status) with
  100% of the filtering logic resolved *before* any multi-GB transfer — satisfying the case's
  "how you inspect and clean geospatial data" criterion cleanly.
- **The catch (see §8)**: knowing *which* patch ids we want does not, by itself, let us pull only
  those patches out of the official `.tar.zst` archives without a full download, unless the
  archives turn out to be internally ordered/seekable in a way we can exploit. This is the
  central open question for a practical access plan.

## 6. Whether official class labels are in metadata

- **Yes.** Each patch carries **scene-level multi-labels** derived from the **CORINE Land Cover
  (CLC) 2018** database, using the **19-class simplified nomenclature** (the "BigEarthNet-19"
  scheme introduced for BigEarthNet in prior work, carried forward as the default for v2.0) —
  as opposed to the original, noisier 43-class CLC nomenclature used in the very first BigEarthNet
  release. **[VERIFIED/REPORTED, consistent across paper abstract + ConfigILM docs]**
- BigEarthNet is inherently **multi-label** (a single patch can carry several CLC classes
  simultaneously) — this is a real modeling consideration for later (single-label subset
  selection vs. genuine multi-label classification), but **is explicitly out of scope to decide
  now** ("Do NOT choose classes yet").
- I could not pull the literal list of the 19 class names from a primary source in this pass;
  the authoritative list is expected inside the dataset description PDF and/or derivable directly
  from the `labels` column of `metadata.parquet` once downloaded. **[UNVERIFIED — needs runtime
  check]** — flagging rather than guessing the list.

## 7. Official geographic splits

- Yes — see §4. The **split is geographic by design** (tile/region-aware, not a naive random
  shuffle), specifically to reduce train/val/test spatial correlation, per the reBEN paper's
  stated contribution. **[VERIFIED, from paper]**
- Recommendation (not yet acted on): use the provided `split` column rather than constructing a
  new split, both to save effort (explicitly out of scope to over-engineer) and to inherit the
  authors' de-correlation guarantees, which we could not easily reproduce ourselves.

## 8. Smallest practical way to obtain ~several thousand paired samples

This is the least-settled question, and worth stating plainly: **there is no official
mechanism to download a spatial/class-based slice of the v2.0 imagery directly.** Concretely:

- The only official distribution channel for paired v2.0 imagery is Zenodo record 10891137, and
  it offers exactly **two monolithic files** — one `.tar.zst` for all of S1 (~54 GB), one for all
  of S2 (~63 GB). There is no per-tile, per-country, or per-split archive offered.
- I checked three plausible "official-adjacent, tile-addressable" alternatives and **none pan
  out for v2.0**:
  - **Microsoft Planetary Computer**: a GitHub discussion *requests* BigEarthNet v2.0 be added as
    a queryable STAC collection, but it has not been fulfilled — v2.0 is **not** on Planetary
    Computer. **[VERIFIED — checked the discussion directly]**
  - **Radiant MLHub**: hosts BigEarthNet, but this is the **v1.0** archive (and Radiant MLHub
    as a service has since been sunset) — not usable for paired v2.0 S1/S2 data.
  - **source.coop/tu-berlin/bigearthnet**: hosted by the same TU Berlin group, but inspection
    shows **590,326 patches from 125 tiles** — that is the **v1.0 Sentinel-2-only** figure, not
    v2.0's 549,488-pair figure. Not paired S1/S2, not v2.0, and not tile-addressable at the level
    I could confirm. Not usable here.
  - **Hugging Face**: the only dataset artifact under the official `BIFOLD-BigEarthNetv2-0` org
    is a small text/label file (`BigEarthNet.txt`, ~9.5 MB) — no image data is mirrored there
    officially. (Third-party HF mirrors like `torchgeo/bigearthnet` exist but their version
    provenance relative to the official v2.0 release is unconfirmed, and the case asked us to
    prefer official sources — so these are not recommended as a primary source.)

- So, practically, **any imagery at all requires reading from one or both of the official
  `.tar.zst` files.** The only lever available to avoid a 100+ GB download is the **internal
  ordering/format of those archives**, which is currently unknown (§1). Two outcomes are
  possible once we actually look:

  1. **If entries can be streamed and filtered on the fly** (i.e., `tar`+`zstd` decompression can
     be piped directly from an HTTP stream and individual member files can be selected by name
     without decompressing the whole archive to disk) — then we can request our chosen patch ids
     from §5, stream through the archive, extract only matching members, and **abort the
     connection once every wanted patch has been seen**. Whether this transfers "a few GB" or
     "most of 54–63 GB" depends entirely on whether matching patches are clustered early in the
     stream or scattered throughout — which in turn depends on the *actual* internal ordering
     (alphabetical-by-S2-product-name is what's reported, and that string starts with mission +
     acquisition date, which would sort **chronologically**, not geographically/by-class — so
     entries for an arbitrary class-based subset may in fact be scattered across the whole file,
     not clustered near the start). This needs a **cheap, safe empirical probe** (a small
     HTTP-range read of the first tens of MB of each archive, enough to read early tar headers)
     before committing to this plan.
  2. **If the archive is not stream-filterable this way** (e.g. proves to be a single opaque
     zstd frame that tooling won't let us seek/filter mid-stream), the only official route to any
     v2.0 imagery is a full ~54–63 GB single-file download — which the case explicitly asks us to
     avoid at 100+ GB scale, but a *single* modality (S1 **or** S2 alone) is technically under
     that bar individually, so a fallback of downloading **one full modality archive** (not both)
     and deriving all three experimental conditions' *optical* need from a further-thinned local
     subset, while capping the *other* modality's transfer by the "clustered early" hope in (1),
     is the worst-case-but-still-bounded plan.

**Bottom line for §8**: the smallest practical path is metadata-first (near-zero cost, resolves
§5/§6/§7 fully offline), followed by a small diagnostic probe of archive internals (cheap,
bounded, safe) to decide between "streamed partial extraction" (best case, likely low single-digit
GB for a few-thousand-patch subset) and "one full modality download, capped partial extraction of
the other" (worst case, ~54–63 GB for one modality — still short of the 100+ GB the case warns
against, though non-trivial). **Both branches stay under the letter and spirit of "do not download
100+ GB," but only the probe tells us which branch we're actually in.**

---

## Estimated disk requirements

| Item | Size |
|---|---|
| `metadata.parquet` + snow/cloud parquet | ~4.3 MB |
| `Reference_Maps.tar.zst` (pixel-level maps, likely not needed for scene classification) | ~282 MB |
| A few-thousand-patch S2 subset (best case, streamed-and-filtered) | rough order: a few hundred MB to low single-digit GB, depending on patch size/band count |
| A few-thousand-patch S1 subset (same caveat) | similar order, somewhat smaller (2 bands vs. up to 12) |
| Worst case: one full modality archive downloaded whole | 54–63 GB (single modality only) |
| Avoid: both full archives | ~118 GB — explicitly what we are trying not to do |

(Per-patch byte sizes are not yet confirmed — GeoTIFF patch dimensions/bit depth would need to be
read from the dataset description PDF or a sample file to make this table exact; the order-of-
magnitude reasoning above is a placeholder pending that.)

---

## Recommended access/download strategy for this take-home

1. **Now (near-zero cost, safe to do immediately on approval)**: download only
   `metadata.parquet` and `metadata_for_patches_with_snow_cloud_or_shadow.parquet` (~4.3 MB
   total) from Zenodo. Inspect schema, split distribution, label distribution, and cloud/snow
   flags. This alone fully answers §5/§6/§7 with certainty instead of secondary-source inference.
2. **Cheap probe (bounded, small)**: issue a small ranged/streamed read (tens of MB, not the
   whole file) against the start of `BigEarthNet-S1.tar.zst` and `BigEarthNet-S2.tar.zst` to
   determine actual internal entry ordering/format, and confirm whether `tar`+`zstd` streaming
   with early-abort filtering is technically viable against these specific files.
3. **Branch based on the probe's result**:
   - If streaming-filter-with-early-abort works: pick a subset from step 1's metadata (stratified
     across a handful of classes and both train/val/test, sized to comfortably fit local
     hardware — exact size/classes deliberately not decided here) and stream-extract just those
     patches from both archives.
   - If it doesn't: fall back to a full download of **one** modality archive at a time (starting
     with whichever is smaller/needed first) and re-evaluate before touching the second, rather
     than queuing both 54 GB and 63 GB transfers up front.
4. Use the **official `split` column** as-is rather than re-deriving train/val/test, to preserve
   the authors' geographic de-correlation guarantee.
5. Defer exact class selection and exact subset size to the next planning stage, as instructed.

---

## Recommendation

**Do metadata-first filtering (step 1) plus the small archive-ordering probe (step 2) now, then
decide the extraction method (step 3) from what the probe actually shows — rather than
pre-committing to either "stream with early-abort" or "full single-modality download."**

Why: every other candidate shortcut (Planetary Computer, Radiant MLHub, source.coop, Hugging
Face image mirrors) turned out on inspection to either not host v2.0 at all, host the wrong
(v1.0, S2-only) data, or not host imagery at all — so there genuinely is no official
lower-effort path around the two Zenodo archives for *paired* v2.0 data. Given that, the only
lever left is the archives' internal structure, which is unknown until we look — and looking is
itself a ~10s-of-MB operation, not a 100+ GB one. This keeps us fully on the official source,
costs almost nothing to attempt, and produces a decision (rather than a guess) about how the
larger extraction should proceed. It also means class/subset selection can be made an informed
choice constrained by what's cheaply retrievable, rather than picked blind and then discovering
it forces a full 118 GB download anyway.

**This document does not authorize any download.** Per your instruction, I am stopping here —
please confirm before I fetch the metadata parquet files or run the archive-ordering probe.

---

---

## Phase 2 — Empirical Findings (metadata downloaded + archive probe run)

Everything in this section was directly verified against the official files — nothing here is
inferred from secondary sources. Method: downloaded `metadata.parquet` and
`metadata_for_patches_with_snow_cloud_or_shadow.parquet` in full (3,616,349 B and 710,162 B —
byte-for-byte matching the Zenodo API's reported file sizes) and loaded both with
pandas/pyarrow; then issued two HTTP range requests (30 MB each, `Range: bytes=0-31457279`,
confirmed via `206 Partial Content` + `Content-Range` + `Accept-Ranges: bytes`) against the start
of `BigEarthNet-S1.tar.zst` and `BigEarthNet-S2.tar.zst`, and streamed each through a Python
`zstandard` decompressor into `tarfile`'s streaming reader to inspect member ordering.

### 1. Confirmed schema (both parquet files, identical columns)

```
patch_id                   str    e.g. "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"
labels                      list   e.g. ["Arable land", "Broad-leaved forest", "Mixed forest", "Pastures"]
split                      str    "train" | "validation" | "test"
country                    str    e.g. "Austria"
s1_name                    str    e.g. "S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57"
s2v1_name                  str    the corresponding BigEarthNet-v1.0 S2 patch name (legacy cross-reference)
contains_seasonal_snow     bool
contains_cloud_or_shadow   bool
```

- `metadata.parquet`: **480,038 rows** (the "clean" patches). `metadata_snow_cloud.parquet`:
  **69,450 rows** (60,773 snow-flagged + 8,677 cloud/shadow-flagged — these two flags are
  mutually exclusive within this file, i.e. no row has both). **Combined: 549,488 rows — an
  exact match to the official total.**
- **Zero nulls, zero duplicate `patch_id`, zero duplicate `s1_name`** across the full 549,488
  rows — pairing is 1:1 and complete; every S2 patch has exactly one named S1 counterpart with no
  missing links. This resolves §2's open question directly: **pairing is a straightforward
  `s1_name` column lookup, nothing to reverse-engineer.**
- `labels` is genuinely multi-label: mean 2.95 labels/patch, min 1, max 11, median 3.
- **19 distinct classes total** — confirms the 19-class nomenclature reported in Phase 1. Full
  list with combined (train+val+test) frequency, most to least common:

  | Class | Patches | Class | Patches |
  |---|---:|---|---:|
  | Mixed forest | 205,952 | Marine waters | 70,301 |
  | Coniferous forest | 201,537 | Urban fabric | 67,661 |
  | Arable land | 200,642 | Agro-forestry areas | 33,206 |
  | Transitional woodland, shrub | 165,511 | Permanent crops | 29,733 |
  | Broad-leaved forest | 145,479 | Inland wetlands | 28,148 |
  | Land principally occupied by agriculture, with significant areas of natural vegetation | 139,036 | Moors, heathland and sclerophyllous vegetation | 14,637 |
  | Complex cultivation patterns | 103,169 | Natural grassland and sparsely vegetated areas | 13,691 |
  | Pastures | 100,130 | Industrial or commercial units | 12,120 |
  | Inland waters | 89,249 | Coastal wetlands | 1,620 |
  | | | Beaches, dunes, sands | 1,351 |

  Every one of the 19 classes appears in **all three splits** with non-zero count (checked
  directly — no class is split-exclusive), and some classes are strongly **country-concentrated**:
  "Agro-forestry areas" is **Portugal-only** (33,206/33,206), "Marine waters" appears in only 4 of
  10 countries, "Coastal wetlands" and "Beaches, dunes, sands" are near-absent outside a handful of
  coastal countries. This is the main practical constraint on class selection (§4 below).

### 2. Confirmed official split counts

| Split | Patches | Share |
|---|---:|---:|
| train | 272,544 | 49.6% |
| validation | 139,577 | 25.4% |
| test | 137,367 | 25.0% |

Roughly a 50/25/25 split, close to balanced between validation and test. Also cross-tabulated
against the clean/affected distinction: of the "affected" (snow/cloud/shadow) 69,450 rows,
34,673 are train / 17,235 validation / 17,542 test — i.e. affected patches are **not**
concentrated in one split, they are spread proportionally, so the official split is safe to use
as-is with either the clean-only file or the combined file.

Country distribution (for context, combined both files): Finland 211,949 · Portugal 89,855 ·
Serbia 76,525 · Ireland 51,705 · Lithuania 51,538 · Austria 45,891 · Belgium 11,314 ·
Switzerland 4,888 · Luxembourg 3,623 · Kosovo 2,200. Coverage is heavily skewed toward Finland
(~39% of all patches) — worth keeping in mind for any country- or tile-based subsetting so a
subset isn't accidentally dominated by one country's land-cover profile.

### 3. Archive internal structure (from the ordering probe — this was the open question in Phase 1)

- **Both `.tar.zst` files are single zstd frames** (checked: the zstd frame magic number
  `28 B5 2F FD` occurs exactly once, at byte 0, in each 30 MB sample). This confirms the
  Phase-1 worry: **there is no seekable/indexed structure** — the only way to reach any content
  partway through either archive is to sequentially decompress from byte 0.
- **Tar entries are grouped hierarchically**: top-level folder (`BigEarthNet-S2/` or
  `BigEarthNet-S1/`) → one subfolder per full **product** (a single Sentinel-2 tile+acquisition-
  date, or a single Sentinel-1 scene) → one subfolder per **patch** (grid row_col) → one GeoTIFF
  per band (12 bands for S2: B01–B12 + B8A; 2 bands for S1: VV, VH).
- **S2 archive order matches ascending alphabetical order of the full product name** — verified
  directly: the first entries decompressed from byte 0 belong to product
  `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP` (Austria), which is exactly the first name when
  all 115 distinct S2 products in the metadata are sorted alphabetically. Since the product name
  starts with mission+acquisition-date, this ordering is effectively **chronological**, not
  grouped by country or class.
- **S1 archive order likewise matches ascending alphabetical order of S1 product names** — but
  because S1 product names start with the satellite id (`S1A_...` vs `S1B_...`), alphabetical
  sort produces **two large contiguous blocks: all 178 `S1A` products first (ranks 0–177), then
  all 134 `S1B` products (ranks 178–311)** — this is a mission split, not a date split, and it
  matters a lot for extraction cost (below).
- Patches per S2 product: mean 4,778, min 95, max 8,269 (115 products total) — so **a single S2
  tile alone typically yields several thousand patches**, already close to the "several thousand"
  target mentioned in the case.

### 4. Selecting several classes while preserving pairing and official splits

Because pairing is a lossless 1:1 metadata join (`s1_name`) and the `split` column already
carries the authors' geographic de-correlation guarantee, **class selection does not risk
breaking pairing or splits** as long as subsetting is done by filtering rows of the already-
joined metadata table (keep `patch_id` + `s1_name` + `split` together) rather than by any
re-derivation. The one real constraint is **which classes/countries are cheap to physically
extract**, given §3's ordering:

- Picking classes/countries **arbitrarily** (e.g. hand-picking 5 "interesting" classes without
  regard to where their patches sit in the archive) could require streaming very deep into
  both archives, since countries and classes are scattered across the full year of acquisitions.
- Picking the **earliest few S2 products in archive order** instead is nearly free on the S2 side
  (a handful of products = tens of thousands of patches for a few hundred MB–few GB) and, per the
  class table below, already yields good class coverage:

  | Leading S2 products (archive order) | Cumulative patches | Classes covered | Countries |
  |---:|---:|---:|---|
  | 1 (T33UUP, Austria) | 3,305 | 14 / 19 | Austria |
  | 2 (+ T34VER, Finland) | 9,991 | 16 / 19 | Austria, Finland |
  | 3 (+ T29UPU, Ireland) | 17,977 | 18 / 19 | + Ireland |
  | 5 (+ T35VPK Finland, T29SND Portugal) | 34,190 | **19 / 19** | + Portugal |

  So **all 19 official classes are reachable from just the first 5 products in archive order** —
  no need to hunt for Kosovo/Belgium/Switzerland-only content, and no need to reshuffle the
  archive-order assumption.
- **The S1 side is the actual bottleneck**, not the S2 side: every one of the first 20 S2 products
  checked pairs with at least one `S1B` scene, forcing a stream across the S1A→S1B boundary
  (rank ~178) regardless of which early tiles are chosen — because BigEarthNet pairs each patch
  with whichever Sentinel-1 pass was temporally closest, and both satellites were flying over
  these tiles within days of each other in mid-2017. Concretely, for the 5-product/19-class
  selection above: 21 distinct S1 products are needed, spanning ranks 0–188 (i.e. 60.6% of the
  312-product S1 list by count, ≈34.8 GB by volume-weighted estimate) — **just to reach content
  that pairs with S2 tiles sitting in the first 4% of the S2 archive.**
- **Refinement that fixes this**: within the chosen S2 tiles, **keep only the patches whose
  paired S1 scene is an `S1A` product** (drop the `S1B`-paired patches from the candidate set).
  This is a legitimate metadata filter (no re-derivation, splits and pairing stay intact) and,
  checked directly on the 5-tile/19-class selection: **13,932 of the 34,190 patches keep their
  `S1A` pairing, still cover all 19/19 classes, and still have all three splits represented**
  (train 6,768 / validation 3,638 / test 3,526) — while the S1 side now only needs to stream
  through **rank 9 of 312** (≈1.4 GB) instead of rank 188 (≈34.8 GB).

**Recommended selection rule**: take the first 5 S2 products in archive order (Austria, Finland ×2,
Ireland, Portugal — 34,190 patches, 19/19 classes, all splits), then filter to the subset of those
patches whose `s1_name` resolves to an `S1A`-prefixed product. Net result: **13,932 paired,
correctly-split, full-class-coverage samples**, requiring an estimated **~4.2 GB streamed from
`BigEarthNet-S2.tar.zst`** (must still read the full 5 tiles' worth of optical patches — the S1A
filter only reduces the S1 side, since S1A/S1B pairing is decided per-patch, not per-tile) **and
~1.4 GB streamed from `BigEarthNet-S1.tar.zst` — roughly 5.6 GB total**, versus the 118 GB of
both full archives. This is a data-driven answer to Phase 1's open §8 question, not a guess: it
was derived from the actual metadata join, not assumed.

(Caveat carried forward honestly: dropping `S1B`-paired patches is a convenience constraint, not
a scientific one — it shifts the sample toward whichever sub-area of each tile happened to be
nearer an S1A pass. This is a reasonable trade-off for a take-home-scale subset, but is worth one
sentence of disclosure in the eventual README as a sampling caveat, not hidden.)

### 5. Updated disk/transfer estimate (replaces the Phase-1 placeholder table)

| Item | Size |
|---|---:|
| Metadata (both parquet files) — **already downloaded** | 4.3 MB |
| Archive-ordering probe (2 × 30 MB range reads) — **already downloaded** | 62 MB |
| Recommended subset: stream `BigEarthNet-S2.tar.zst` through the 5th target product | ~4.2 GB |
| Recommended subset: stream `BigEarthNet-S1.tar.zst` through S1A rank 9 | ~1.4 GB |
| **Total for 13,932 paired, 19-class, all-split samples** | **~5.6 GB** |
| (for reference) one full modality archive | 54.4–63.3 GB |
| (for reference) both full archives | ~118 GB — not needed |

### 6. What this means for the next stage (not done here)

Extraction itself (actually streaming ~5.6 GB and writing the ~13,932 matched patch folders to
disk) has **not** been performed — this pass only downloaded metadata (4.3 MB) and two 30 MB
ordering probes (62 MB total), per the instruction to keep this step small. The next stage would
be to implement the streaming-extract-and-early-abort script using the exact member-ordering
behavior confirmed above, still without training anything.

---

## Sources consulted

- [bigearth.net](https://bigearth.net/) — official project site and downloads section
- [Zenodo record 10891137](https://zenodo.org/records/10891137) — official v2.0 data host, file listing/sizes
- [reBEN paper, arXiv:2407.03653](https://arxiv.org/abs/2407.03653) — Clasen et al. 2024, dataset construction and split methodology
- [ConfigILM BigEarthNet v2.0 docs](https://lhackel-tub.github.io/ConfigILM/extra/DataSets%20and%20DataModules/bigearthnetv2.html) and [API reference](https://lhackel-tub.github.io/ConfigILM/API/ds/api_ds_BENv2.html) — loader/metadata usage details
- [rsim-tu-berlin/bigearthnet-pipeline](https://github.com/rsim-tu-berlin/bigearthnet-pipeline) — official dataset-construction pipeline, mapping-file structure
- [microsoft/PlanetaryComputer discussion #447](https://github.com/microsoft/PlanetaryComputer/discussions/447) — confirms v2.0 is *not* on Planetary Computer
- [source.coop/tu-berlin/bigearthnet](https://source.coop/tu-berlin/bigearthnet) — confirmed this is v1.0 S2-only, not usable
- [BIFOLD-BigEarthNetv2-0 org on Hugging Face](https://huggingface.co/BIFOLD-BigEarthNetv2-0) — confirmed no official imagery mirror, only labels file and pretrained models
- [wri/ben_labels](https://github.com/wri/ben_labels) — background on original 43-class nomenclature (for contrast with v2.0's 19-class scheme)
