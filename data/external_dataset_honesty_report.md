# Is Our "Clinical" External Dataset Really Clinical Data? — An Honest Assessment

*Project: hERG Channel Blockade (Novartis Discoverathon, Challenge 2 · Safety)*
*Scope: DICTrank + CredibleMeds QTdrugs, as used for external validation of our ChEMBL-trained hERG model*

---

## TL;DR — the honest verdict

**No — not "clinical data" in the strict sense.** Neither DICTrank nor CredibleMeds is raw, patient-level clinical measurement (no ECGs, no per-patient QT intervals, no trial endpoints, no doses). Both are **expert- and regulator-curated *reference labels*, assigned at the level of the whole drug**, that *summarize* accumulated clinical evidence into a small set of ordinal risk categories.

The accurate term for what we have is **"clinically-derived cardiotoxicity annotations"** or **"regulatory/expert-adjudicated risk labels"** — a distilled *proxy* for clinical cardiotoxicity, not clinical data itself. This distinction is not pedantic: it changes what we are allowed to claim, and it is exactly the kind of rigor the rubric's 35% "methodology and data handling" weight rewards. Used honestly — as a coarse, directional **external check** rather than a ground truth — the datasets are genuinely valuable. Used loosely — described as "clinical validation" or folded into training — they would misrepresent the work.

---

## 1. What "clinical data" would actually mean

If a reviewer hears "we validated against clinical data," they will reasonably picture measurements taken from patients: recorded QT intervals from ECGs, thorough-QT (TQT) study results, per-patient torsades events with the dose and plasma concentration at which they occurred, and the covariates (electrolytes, co-medications, genetics) that modulate them. That is measurement-level, per-patient, per-exposure data.

What we actually hold is one row per *drug* with a single categorical label. Everything that made that label — the patients, the doses, the ECGs, the case reports — has already been read, weighed, and collapsed by a human committee into a word ("most concern", "Known Risk"). We have the **conclusion of a clinical evidence review, not the clinical evidence.**

## 2. What DICTrank actually is (grounded in the source paper)

The DICTrank Predictor paper we are working from (Seal et al., *J. Chem. Inf. Model.* 2024) states plainly how the labels were made: the DICTrank set "was generated from an expert review from the FDA, keyword searches, and manual curation of FDA labeling documents as well as data from clinical trials, postmarketing, and literature surveys," categorizing 1,318 drugs into four DICT-Concern categories (most / less / no / ambiguous) by their *potential risk* for cardiotoxicity.

Three honest consequences follow directly:

The labels are **FDA-drug-label-derived and categorical**, not measured. A drug is "most concern" because its label text and the reviewers' reading of the evidence put it there — a regulatory-linguistic signal, not a number from a patient.

The same paper is candid that this is a proxy: "predicting any in vivo effect is not a trivial classification task, and most predictive models are built on proxy or reduced end points (which are often reduced to binary end points) **without taking into account in vivo parameters such as pharmacokinetic parameters.**" In other words, the endpoint deliberately discards dose and exposure — the very things that determine whether hERG binding becomes a clinical event.

The **"no-concern" label means "evidence of absence" of a cardiotox signal in the label**, not "proven cardiac-safe." (Table 1 of the paper literally annotates the non-toxic class as "evidence of absence.") That makes our negative class asymmetrically noisy: some "no-concern" drugs are simply under-studied or newer, not truly safe.

## 3. What CredibleMeds QTdrugs actually is (grounded in their methodology)

CredibleMeds (run by the non-profit AZCERT) assigns its Known / Possible / Conditional Risk categories using a documented process called **ADECA™ (Adverse Drug Event Causality Analysis)**, which applies Bradford-Hill causality criteria to a body of *published* evidence: PubMed literature, FDA labels, FDA FAERS/AERS adverse-event signals, reports submitted to CredibleMeds, and laboratory/clinical experiments. A formal evidence analysis is written up and **adjudicated at a weekly expert committee meeting**; disagreement escalates to a 34-member advisory board.

So CredibleMeds is, even more explicitly than DICTrank, **expert adjudication of the clinical literature** — a curated verdict, drug-level, categorical. Two of their own stated limitations matter for us: FAERS data "lacks overall drug exposure measures, preventing risk-ratio calculations" (again, no dose/exposure), and the Conditional Risk tier exists precisely because those drugs' risk is *confounded* by drug interactions, overdose, or electrolyte abnormalities. CredibleMeds themselves warn against using the list to rank-order drugs for relative toxicity.

## 4. The uncomfortable one: partial circularity with hERG

This is the point most easily glossed over, so we state it directly. **hERG data is itself an input to how these "clinical" labels are assigned.** CredibleMeds' ADECA process lists "hERG channel blockade data" among its 16 evidence categories, and regulatory cardiotox assessment (the basis of DICTrank) has been organized around hERG/QT liability for two decades (the ICH S7B / E14 framework). The DICTrank paper's own headline finding — that predicted KCNH2/hERG activity best separates the concern categories — is therefore *partly expected by construction*: a drug's known hERG liability helped put it in its category in the first place.

This does **not** make our external check worthless, but it means we cannot claim it as a fully independent confirmation that "hERG structure predicts clinical outcomes." Part of the correlation we measure is the mechanism genuinely propagating to the clinic; part is the label having been informed by hERG evidence to begin with. Honest reporting acknowledges both.

## 5. Caveats introduced by *our own* processing

Beyond the sources, our pipeline added its own, smaller layer of approximation, and the report should own these too:

- **Name → structure mapping.** CredibleMeds ships drug *names only*. We resolved SMILES offline against DrugBank / ChEMBL-MOA / EPA CompTox / LINCS tables (PubChem was network-blocked), standardizing with RDKit. Of 112 CredibleMeds-only drugs, **102 resolved; 10 did not** — 2 combination products, 4 biologics (antibodies/peptides that have no small-molecule structure and are out of scope for a SMILES model anyway), and 4 post-2023 small molecules absent from the offline tables. Name-based resolution can occasionally map to the wrong salt/parent or a synonym.
- **Salt/parent standardization.** We reduced each structure to its largest organic fragment and keyed matches on the InChIKey connectivity block, which is robust but deliberately ignores stereochemistry and protonation state.
- **Drug-level, so no replication or uncertainty.** Each clinical row is a single categorical verdict with no error bar, unlike our ChEMBL pIC50 values which carry measurement spread.

## 6. What this means for how we use it (and how we word it)

The datasets are **appropriate as a coarse, directional external validation signal, and inappropriate as training labels** — the same conclusion we reached earlier, now fully justified. Our own premise check bears out both halves: on the drugs that overlap our ChEMBL data, hERG blocker rate rises monotonically with DICTrank concern (no 42% → less 53% → most 71%; Known-Risk TdP drugs 88%), but the separation AUC is only ~0.65 — a real but **modest** signal, exactly what a *downstream, exposure-confounded, partially-circular* proxy should produce. If we saw AUC ~0.95 we should be *suspicious*, not pleased.

Concretely, for the write-up:

**Claims we can honestly make:** "Our hERG model's scores *enrich for* drugs that regulators and clinical experts have flagged as cardiotoxic," and "the model *generalizes* from ChEMBL bioactivity space to clinically-annotated marketed drugs it never saw in training (957 external compounds, held out by InChIKey)."

**Claims we must avoid:** "Our model predicts clinical cardiotoxicity / QT prolongation / torsades," or "validated on clinical data." We predict *hERG blockade*; DICTrank/CredibleMeds record an expert *risk verdict* several causal steps downstream.

**One more honesty flag — chemical-space bias.** These are overwhelmingly *approved, marketed* drugs: drug-like, orally viable, already past safety gauntlets. That is a narrower, biased slice of chemistry than the ChEMBL screening compounds our model trains on, so strong enrichment here does not guarantee the model generalizes to novel or non-drug-like chemistry. Say so.

---

## Bottom line

Call it what it is: **a clinically-*informed*, expert-curated external reference set**, not clinical data. Framed that way, it is one of the strongest parts of the submission — it shows the model's preclinical predictions track real-world cardiac risk — precisely *because* we are honest about its being a coarse, confounded, partially-circular proxy rather than overselling it as ground truth.

---

### Sources

- Seal, S., Spjuth, O., Hosseini-Gerami, L., García-Ortegón, M., Singh, S., Bender, A., Carpenter, A. E. *Insights into Drug Cardiotoxicity from Biological and Chemical Data: The First Public Classifiers for FDA Drug-Induced Cardiotoxicity Rank.* J. Chem. Inf. Model. 2024, 64, 1172–1186. (DICTrank provenance, proxy-endpoint and Cmax caveats, Table 1 "evidence of absence".)
- DICTrank dataset — U.S. FDA, Drug-Induced Cardiotoxicity Rank (2023); 1,318 drugs, four DICT-Concern categories from FDA labeling.
- CredibleMeds® / AZCERT — QTdrugs List and the ADECA™ (Adverse Drug Event Causality Analysis) process for evaluating evidence and assigning TdP risk (Known / Possible / Conditional); QTdrugs list rev. 16 April 2026. https://crediblemeds.org
- Our pipeline: `clinical_cardiotox_combined.csv`, `clinical_external_validation_set.csv`, and the overlap/premise-check analysis over the training `Data.zip` (16,650 ChEMBL compounds; 957 external clinical drugs; premise-check AUC ≈ 0.65).
