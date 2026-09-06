# Reactor Yield Prediction: Fugacity ML Hackathon

A physics-informed hybrid model for predicting overall yield in a non-isothermal continuous-flow chemical reactor, built for the ML Hackathon at Fugacity, the annual fest of the Department of Chemical Engineering, IIT Kharagpur.

**Team:** mittalakshat43
**Members:** Akshat Mittal, Aishee Lahiri, Divesh Dangare

---

## Problem

The reactor follows A to B to C consecutive first-order kinetics. Given five operating parameters:

- `flow_rate_L_min`
- `concentration_mol_L`
- `inlet_temperature_K`
- `length_m`
- `jacket_temperature_K`

the task is to predict `overall_yield`.

The competition ran in two phases: a pure RMSE leaderboard (Phase 1), and an offline judging round requiring a presentation defending the model with physical reasoning (Phase 2). Our team advanced to Round 2.

## Approach

We built a physics-informed hybrid model rather than treating this as a pure black-box regression problem, since the underlying system is a well-understood chemical engineering process.

**1. Mechanistic sub-model**
A kinetics and heat-transfer model combining:
- NTU (Number of Transfer Units) heat-transfer relations
- A closed-form solution for consecutive first-order reaction kinetics

Parameters were fit using `scipy.curve_fit` with multiple random restarts to avoid local minima, using tight convergence tolerances (`xtol = ftol = gtol = 1e-14`) for deterministic results.

**2. Feature engineering**
Outputs and intermediate quantities from the mechanistic model were used as engineered features, rather than feeding raw operating parameters directly into the ML layer.

Two notable modeling decisions:
- `concentration_mol_L` is deliberately excluded from the mechanistic yield formula. For first-order kinetics, the initial concentration mathematically cancels out of the yield ratio, confirmed empirically on this data.
- Residence time is computed as `tau = length_m / flow_rate_L_min`, which assumes constant reactor cross-section. This is a documented simplifying assumption.

**3. ML layer**
A Bagged MLP ensemble (`sklearn` `MLPRegressor` wrapped in `BaggingRegressor`) was trained on the mechanistic features.

We benchmarked this against Ridge Regression, Random Forest, Gradient Boosting, XGBoost, and Gaussian Process Regression, all trained on identical physics-informed features and identical cross-validation splits. The hybrid Bagged MLP outperformed every alternative by a substantial margin on internal CV.

## Results

- **Relative performance:** Outperformed five alternative model families (Ridge, Random Forest, Gradient Boosting, XGBoost, Gaussian Process) by a substantial margin on identical physics-informed features and identical CV splits.
- **Exact metrics:** Leaderboard RMSE: 14.8739. Internal 5-fold CV RMSE: 8.519 ± 2.659 (mean ± std, computed entirely on training data, test set never touched during model selection).
- **Progression:** Advanced from Phase 1 (leaderboard) to the finals (offline judging round)

## Known limitations

- The Bagged MLP stage closely interpolates the training set, which makes final predictions sensitive to floating-point-level differences across numpy, scipy, and BLAS versions. Running the notebook on a different machine can produce a slightly different output CSV even with the same code and seed.
- There is a gap between internal cross-validation RMSE and actual leaderboard RMSE. We investigated this with repeated-holdout resampling and nearest-neighbor distribution checks, and attribute it primarily to distribution shift between train and test sets rather than simple overfitting.

Full analysis of both points is in the Phase 2 presentation slides.

## Acknowledgments

Department of Chemical Engineering, IIT Kharagpur, for hosting this problem as part of Fugacity.
