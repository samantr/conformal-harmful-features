# 3 Method

## 3.1 Problem formulation and analysis target

We study multiclass classifiers whose predictive probabilities are converted into adaptive conformal prediction sets. Let \(X\in\mathbb{R}^d\) denote the input features, \(Y\in\{1,\ldots,K\}\) the class label, and \(S\subseteq\{1,\ldots,d\}\) a retained feature subset. For every candidate subset, the classifier is refit using only the features in \(S\); the proposed method is therefore a retraining-based feature-selection procedure rather than inference-time masking.

The target of selection is not conventional predictive importance. Instead, the method asks whether removing a feature can reduce adaptive conformal prediction-set size while respecting prespecified safeguards on classification accuracy and empirical marginal coverage. Accordingly, conformal harm is defined relative to a complete learning and uncertainty-quantification pipeline: dataset, model family, feature subset, probability scaling method, conformal score, and significance level. The method does not assume that a feature has an intrinsic harmfulness independent of this context.

All feature-selection decisions are made without using the final conformal-calibration or test partitions. Data are divided into four disjoint outer partitions: training, tuning/selection, conformal calibration, and test. The outer training partition is used to fit candidate classifiers and their preprocessing; the outer tuning partition is used to estimate conformal harm and choose the subset; after the subset is frozen, the outer conformal-calibration partition supplies a fresh conformal threshold and the test partition is used once for final evaluation. This separation is designed to prevent calibration- or test-driven subset selection. It does not, by itself, resolve exchangeability concerns in grouped or shifted data such as HAR.

## 3.2 Adaptive conformal prediction and probability scaling

For a fitted classifier, let \(\pi_k(x;T)\) be the softmax probability for class \(k\) after temperature scaling by \(T>0\). Base prediction uses \(T=1\). Ordinary temperature scaling chooses \(T\) on tuning data by minimizing negative log-likelihood [@guo2017calibration]. Conformal Temperature Scaling (ConfTS) instead chooses a temperature using the conformal efficiency-gap objective of Xi et al. [@xi2025confts]. For each candidate temperature, the tuning data available to ConfTS are split into disjoint threshold and loss subsets. A non-randomized APS or RAPS threshold is computed on the threshold subset, and the mean squared difference between that threshold and the true-label non-randomized score is evaluated on the loss subset. The temperature with minimum loss is retained. Final conformal calibration is never used to choose ordinary TS or ConfTS temperatures.

Final APS [@romano2020adaptive] and RAPS [@angelopoulos2021raps] calibration and prediction use randomized scores. For an observation whose class probabilities are sorted in decreasing order, the score of a candidate label at rank \(r\) is

\[
S_{\mathrm{APS}}(x,y)=\sum_{j<r}\pi_{(j)}(x)+U\,\pi_{(r)}(x),
\]

where \(U\sim\mathrm{Uniform}(0,1)\). RAPS adds the regularization term

\[
S_{\mathrm{RAPS}}(x,y)=S_{\mathrm{APS}}(x,y)
+\lambda\max(r-k_{\mathrm{reg}},0).
\]

One uniform variate is shared across the candidate labels of a sample. Within a dataset/seed unit, the same prespecified calibration and test uniform vectors are reused across matched feature-subset comparisons so that randomization does not introduce avoidable between-method noise.

Given \(n_{\mathrm{cal}}\) true-label scores on the outer conformal-calibration partition, the final threshold is the finite-sample split-conformal quantile at rank

\[
\min\left\{n_{\mathrm{cal}},\left\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\right\rceil\right\}.
\]

A candidate label is included when its randomized score is at most this threshold. We report empirical coverage together with mean, median, and 90th-percentile set size and empty/full-set rates. Smaller sets are interpreted as greater efficiency only in conjunction with the corresponding coverage and reliability diagnostics.

## 3.3 Operational definition of conformal harm

The primary formulation is constrained efficiency. For a full-feature reference model \(0\) and a candidate intervention \(S\), tuning-only evidence defines

\[
\begin{aligned}
G_{\mathrm{eff}}(S) &= \overline{|C_0(X)|}-\overline{|C_S(X)|},\\
L_{\mathrm{acc}}(S) &= \mathrm{Acc}_0-\mathrm{Acc}_S,\\
D_{\mathrm{cov}}(S) &= \max\{0,(1-\alpha)-\mathrm{Cov}_S\},\\
V_{\mathrm{cond}}(S) &= \max\{\mathrm{SSCV}_S, D_{\mathrm{class},S}\},
\end{aligned}
\]

where \(D_{\mathrm{class},S}\) is the maximum absolute class-conditional coverage deviation from the target. Positive \(G_{\mathrm{eff}}\) means that removal yields smaller prediction sets. Coverage shortfall is one-sided: empirical overcoverage is not counted as a validity failure because its efficiency cost is already reflected in set size.

Under the frozen primary constraints, a candidate is eligible only if its maximum observed tuning accuracy loss does not exceed 0.01 and its maximum observed tuning coverage shortfall does not exceed 0.03. Among eligible candidates, the proposed method prioritizes prediction-set efficiency. A feature or subset is not labeled conformally harmful merely because it satisfies the safeguards: a positive estimated efficiency gain is also required for the primary harmful-feature interpretation.

Two alternative formulations were retained as supporting analyses during method development but were not used to define the Phase 8 primary selection. The weighted score was

\[
H_{\mathrm{weighted}}=G_{\mathrm{eff}}
-4L_{\mathrm{acc}}-10D_{\mathrm{cov}}-V_{\mathrm{cond}},
\]

with all weights fixed before final calibration or test evaluation. A Pareto formulation separately minimized accuracy loss, prediction-set size, and conditional violation after excluding candidates beyond the coverage-shortfall limit. The constrained formulation was frozen as primary because it directly exposes the accuracy and coverage gates rather than allowing them to be traded away by a scalar score.

## 3.4 Cross-fitted tuning evidence

Candidate harmfulness is estimated wholly inside the outer tuning partition. For every feature subset and model, the classifier and standardizer are fitted on the outer training data. The resulting classifier is then evaluated through five seeded repetitions of four-fold cross-fitting within outer tune. In each repetition, one fold is used as selection-evaluation data. The remaining tuning rows are split, stratified by class, into 60% scaling-tuning and 40% selection-calibration subsets. Scaling parameters are chosen on the scaling-tuning rows, an inner conformal threshold is computed on the selection-calibration rows, and the held-out fold supplies the efficiency, accuracy, coverage, and conditional-reliability measurements. Every outer-tuning row is used exactly once as an evaluation row per repetition.

The primary selection pipeline is Base APS at \(\alpha=0.10\), so the progressive feature order is not optimized over multiple scaling or score choices. Within each cross-fitting repetition, fold-level quantities are averaged and then paired with the corresponding full-feature reference. Eligibility is conservative across repetitions: the maximum accuracy loss and maximum coverage shortfall over the five repeated cross-fitted estimates must satisfy the frozen constraints. The mean conditional violation is retained as a tie-breaker rather than being allowed to compensate for a failed accuracy or coverage gate.

Randomized APS uniforms used in this inner selection protocol are deterministically paired across the full-feature reference and candidate subsets. Thus, the selection signal is based on differences under matched randomization rather than unrelated Monte Carlo draws.

## 3.5 One-shot and recursive harmful-feature removal

We evaluate two prespecified progressive-selection paths. The **one-shot** path first evaluates every single-feature removal from the full feature set. Eligible candidates are ordered primarily by decreasing incremental efficiency gain, followed by lower accuracy loss, lower coverage shortfall, lower conditional violation, and a deterministic feature-name tie-break. That initial single-feature order is then kept fixed while progressively removing features.

The **recursive** path re-evaluates the remaining one-feature removals after every accepted step. At each stage, candidate subsets are refit and assessed with the same cross-fitted tuning protocol. The next feature is the eligible candidate with the largest positive incremental reduction in prediction-set size. In the primary stopping logic, recursive selection stops when no safe positive-increment candidate remains.

For both paths, step zero—the all-feature model—is an admissible choice. Among eligible path steps, subset size is chosen by maximum cumulative tuning efficiency gain; lower conditional violation and then fewer removals break ties. Consequently, the method can select no intervention when the tuning evidence does not support removal.

Phase 8 additionally materialized the progressive paths through up to five removed features for the prespecified subset-size sensitivity analysis, even when the primary stopping rule would have terminated earlier. These diagnostic continuation steps did not replace the primary subset: primary selection still used the tuning-only eligibility, nonnegative-efficiency, and step-zero fallback rules. The same logic was repeated under allowed tuning accuracy-loss values of 0, 0.005, 0.01, and 0.02 for the frozen sensitivity analysis.

## 3.6 Freezing the subset and final evaluation

Once a subset is selected, all feature identities and hyperparameter choices are frozen before final conformal calibration and test evaluation. The classifier is then refit from scratch on the complete outer training partition using only the retained features. Probability scaling is selected on the complete outer tuning partition, a fresh APS/RAPS threshold is obtained from the untouched outer conformal-calibration partition, and the frozen pipeline is evaluated once on the outer test partition. No final calibration or test metric is used to revise the subset, temperature, stopping point, or removal order.

This final refitting step is essential to the estimand: the study measures the effect of retraining a classifier on a reduced feature set rather than perturbing features of a fixed fitted model. Every stored Phase 8 final-result row records that the classifier was refit and that neither final calibration nor test data were used for feature selection. The all-feature model is evaluated under the same split, model seed, scaling choices, conformal settings, and randomized APS/RAPS uniforms, yielding paired within-seed comparisons.

The method therefore targets a conditional, pipeline-specific notion of conformal harm: a feature is a useful removal candidate when tuning-only evidence indicates that excluding it can reduce adaptive prediction-set size while remaining inside the prespecified accuracy and empirical-coverage safeguards. The procedure does not claim that the same feature must be harmful across seeds, models, datasets, or conformal settings, and final empirical coverage remains an outcome to be checked rather than a property enforced by the selector.
