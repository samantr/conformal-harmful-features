# Phase 9 literature and novelty verification

**Search date:** 2026-09-11  
**Purpose:** constrain the Introduction and Related Work to defensible novelty claims before manuscript submission.  
**Status:** manuscript-support record; it does not modify the frozen scientific design or any Phase 8 evidence.

## Scope and source policy

The review focused on work most likely to constrain the manuscript's positioning: conformal classification and adaptive prediction sets; probability calibration and conformal efficiency; conformal-aware feature selection; conventional feature selection and feature importance; feature-selection stability; and conformal validity under grouped or shifted data. Publisher, journal, proceedings, OpenReview, PMLR, JMLR, and arXiv records were prioritized over secondary summaries. Searches included exact titles and combinations of *conformal prediction*, *feature selection*, *recursive feature elimination*, *APS*, *RAPS*, *prediction-set efficiency*, *temperature scaling*, *stability*, *group coverage*, and *distribution shift*.

This is a targeted novelty verification rather than a formal systematic review. Failure to identify an identical method in these searches is therefore **not** evidence that no such work exists. The manuscript should avoid “first”, “only”, or equivalent priority claims unless a later systematic search establishes them.

## Closest prior work and its implications

### Adaptive conformal classification and efficiency

Conformal prediction provides finite-sample marginal coverage under the relevant exchangeability conditions [@shafer2008tutorial]. Adaptive Prediction Sets (APS) introduced a probability-ranking score designed to make multiclass prediction sets adapt to example difficulty [@romano2020adaptive]. Regularized Adaptive Prediction Sets (RAPS) regularized the adaptive score to reduce unnecessary labels, particularly in large-class problems [@angelopoulos2021raps]. These papers establish that prediction-set size is a central efficiency criterion, so the present manuscript cannot present efficiency itself as a new conformal objective.

Several later methods explicitly optimize conformal efficiency at other points in the learning pipeline. Conformal Training differentiates through conformal prediction during model fitting [@stutz2022conftr]. C-Adapter modifies classifier outputs through an order-preserving adapter to improve conformal efficiency while retaining the output ranking [@liu2025cadapter]. Decoupled Conformal Optimisation (DCO) uses independent tuning and fresh calibration data for efficiency-oriented structural selection [@wu2026dco]. Consequently, neither “optimizing conformal efficiency” nor the general train/tune/calibrate separation is unique to the present work.

### Calibration and ConfTS

Temperature scaling is a standard post-hoc confidence-calibration method selected by validation negative log-likelihood [@guo2017calibration]. Xi et al. showed that confidence-calibration objectives and adaptive-conformal efficiency can diverge: ordinary calibration may enlarge adaptive prediction sets, while their Conformal Temperature Scaling (ConfTS) directly tunes the temperature for conformal efficiency [@xi2025confts]. The present study builds on that observation by asking whether a different intervention—removing and retraining on selected features—can improve efficiency, and by examining how that intervention interacts with Base, TS, and ConfTS. The manuscript therefore must describe ConfTS as prior work, not as part of the paper's methodological novelty.

### Conformal feature selection predates this study

The literature contains multiple explicit uses of conformal prediction for feature selection. Bellotti, Luo, and Gammerman proposed Strangeness Minimisation Feature Selection for confidence machines in 2006 [@bellotti2006smfs]. Yang et al. used conformal confidence as a criterion for selecting enough features to reach a desired average confidence level [@yang2011conformalfs], and the later conformal-prediction monograph reviewed feature-selection procedures and their selection-bias/exchangeability implications [@bellotti2014feature]. Zhou et al. proposed conformal feature-selection wrappers for instance transfer (CFSWIT), jointly selecting features and source instances [@zhou2018cfswit].

Most importantly for the present manuscript, López-De-Castro, García-Galindo, and Armañanzas introduced **Conformal Recursive Feature Elimination (CRFE)**, now published in *Pattern Recognition* [@lopezdecastro2026crfe]. CRFE recursively removes features that increase global nonconformity under additively decomposable nonconformity functions, refitting after each removal and providing a data-driven stopping criterion. Its reported evaluation includes prediction-set efficiency and feature-subset stability. CRFE is therefore a close conceptual predecessor.

These sources rule out any claim that the present method is the first conformal feature-selection method, the first recursive feature-elimination method based on conformal prediction, or the first conformal feature selector concerned with prediction-set efficiency.

## Defensible methodological distinction

The manuscript can instead position the contribution narrowly and operationally. The proposed method directly estimates the effect of **retraining after feature removal on modern adaptive conformal set size**, using APS as the primary selection score, while imposing explicit tuning-only safeguards on classification accuracy and empirical marginal-coverage shortfall. Harm is treated as a property of a feature--model--pipeline combination rather than as an intrinsic feature label. Both a fixed one-shot order and a recursively re-estimated removal path are studied, with step zero retained as a valid no-removal choice.

Relative to CRFE, the distinction is therefore not simply “conformal recursive feature elimination.” CRFE derives feature relevance from additive decomposition of a nonconformity function and recursively removes the feature with the largest nonconformity contribution. The present method uses retrained ablations and directly measures adaptive prediction-set efficiency, accuracy loss, marginal-coverage shortfall, and conditional-reliability diagnostics on tuning-only cross-fitted evidence. It then freezes the subset before a fresh outer conformal calibration and test evaluation. This difference should be stated as a methodological design choice, not as proof of superiority.

The experimental contribution is also relevant to positioning: the frozen study evaluates two downstream model families, three real datasets, conventional selectors including MI, permutation importance, RFE, SHAP, and CRFE, multi-seed feature-identity stability, APS/RAPS sensitivity, Base/TS/ConfTS interactions, subject-disjoint HAR behavior, subject-level clustered uncertainty, and numerical boundary diagnostics. These evaluation dimensions support the paper's claim of a broader robustness study, but they do not make the method universally effective.

## Conventional importance is not the same estimand

Classical feature selection is usually motivated by predictive performance, parsimony, computational cost, or interpretability [@guyon2003feature]. RFE recursively eliminates features using a predictive-model relevance criterion [@guyon2002rfe]; SHAP assigns feature contributions to model outputs [@lundberg2017shap]; and permutation-based importance measures the dependence of predictive performance on a feature, with that dependence itself varying across well-performing models [@fisher2019reliance]. These objectives need not coincide with the effect of removing a feature on the size of a calibrated adaptive prediction set.

This supports the manuscript's conceptual distinction between **predictive importance** and **conformal-efficiency contribution**. It does not imply that conventional importance is inferior in general, nor does the observed disagreement establish a universally stable identity of harmful features.

## Stability and feature identity

Feature-selection stability concerns the robustness of selected features or rankings to sampling and algorithmic variation [@nogueira2018stability]. Kuncheva's chance-corrected overlap index is one established measure of subset stability [@kuncheva2007stability]. Conformal methods have also been proposed specifically for uncertainty quantification of feature-selection stability estimates [@lopezdecastro2024conformalstability].

The manuscript should therefore distinguish two questions: whether a selection procedure produces a reproducible **performance effect**, and whether it selects the same **feature identities** across seeds. The frozen evidence supports cross-method disagreement in selected removals but shows weak across-seed identity stability in several settings, particularly the small neural network. This should be presented as a limitation of attribution, not hidden by aggregate efficiency effects.

## Grouped data, shift, and the HAR boundary

Standard conformal guarantees rely on the relevant exchangeability conditions. Barber et al. explicitly study the degradation and modification of conformal guarantees beyond exchangeability [@barber2023beyond], while newer work develops richer group-conditional coverage objectives [@bairaktari2025kandinsky]. These lines of work make the HAR wording especially important.

The subject-disjoint HAR split is an experimental safeguard against direct subject leakage. It is **not** evidence that windows from held-out subjects are exchangeable with calibration windows, and it does not guarantee nominal 0.90 coverage. Subject-level bootstrap evidence in this paper quantifies robustness of observed effects over the sampled seed/subject structure; it should not be presented as a replacement conformal-coverage theorem.

## Novelty wording allowed and wording to avoid

Defensible phrasing includes statements such as: “we study an efficiency-oriented feature-selection formulation for adaptive conformal classification”; “we define conformal harm operationally through retrained feature-removal effects on prediction-set size subject to explicit accuracy and coverage safeguards”; and “we evaluate whether these effects persist across models, datasets, conventional selectors, probability scaling, seeds, and subject-disjoint HAR splits.” These claims describe what the study actually does without asserting historical priority.

Avoid the following formulations in the manuscript: “the first conformal feature-selection method”; “the first recursive conformal feature selector”; “the first method to optimize conformal set size using feature selection”; “the first method to separate tuning from calibration for conformal optimization”; “harmful features are intrinsically different from important features”; or “subject-disjoint splitting guarantees conformal validity.” Existing literature directly contradicts some of these claims, and the frozen evidence is insufficient for the others.

## Result claims that literature positioning must not broaden

Literature context does not change the verified H1--H4 decisions. H1 remains supported with qualification: the selected removals disagree with conventional selectors, but feature identity is not universally stable. H2 remains partially supported overall, with the strongest evidence on Dry Bean small NN; HAR efficiency gains occur under subnominal coverage. H3 remains partially supported descriptively as **sub-additive complementarity**, not general synergy. H4 remains not generally supported. No related-work comparison should be used to convert these empirical boundaries into stronger claims.

## Literature status before submission

As of 2026-09-11, the closest identified feature-selection paper is CRFE, alongside earlier SMFS/conformal-confidence feature selection and CFSWIT. Contemporary efficiency work such as ConfTr, ConfTS, C-Adapter, and DCO shows that set-efficiency optimization and tune/calibrate decoupling are active and non-unique design ideas. The manuscript's defensible contribution is therefore the specific conformal-harm formulation and its robustness analysis rather than a broad historical-priority claim.

A final pre-submission search should be repeated because 2026 literature is moving quickly, especially for conformal optimization and conformal feature selection. If a new close method appears, the novelty paragraph should be narrowed further rather than altering the frozen experiments.
