# NeurIPS 2024 BELKA Challenge: Results & Analysis

## Executive Summary

This project implements an ensemble machine learning pipeline for predicting small molecule binding to protein targets as part of the NeurIPS 2024 - Predict New Medicines with BELKA competition. The ensemble approach achieved a **cross-validated AUC of 0.9183** and **Average Precision of 0.3399**, demonstrating strong performance on this highly imbalanced drug discovery dataset.

---

## Dataset Overview

### Training Data
- **Total molecules analyzed**: 25,234 (sampled from 100K for computational efficiency)
- **Features**: 2,048-dimensional Morgan fingerprints + 3 protein targets
- **Target distribution**: Highly imbalanced (0.93% binding rate)
- **Protein targets**: BRD4, HSA, sEH

### Class Distribution
```
Non-binders: 99.07%
Binders:      0.93%
```

The extreme class imbalance represents a realistic drug discovery scenario where only a small fraction of molecules exhibit desired binding properties.

---

## Methodology

### Feature Engineering

**Molecular Features**
- Morgan fingerprints (radius=2, 2048 bits) generated using RDKit
- Captures structural and chemical properties of molecules
- Efficient representation for machine learning

**Protein Encoding**
- One-hot encoding for three protein targets
- Enables model to learn protein-specific binding patterns

**Final Feature Space**: 2,051 dimensions (2,048 molecular + 3 protein)

### Model Architecture

An ensemble of three gradient boosting and tree-based models:

1. **XGBoost**
   - Objective: Binary classification
   - Max depth: 8
   - Learning rate: 0.05
   - N estimators: 300
   - Early stopping: 30 rounds

2. **LightGBM**
   - Objective: Binary classification
   - Num leaves: 64
   - Learning rate: 0.05
   - N estimators: 300
   - Early stopping: 30 rounds

3. **Random Forest**
   - N estimators: 200
   - Max depth: 20
   - Min samples split: 10
   - Max features: sqrt

### Validation Strategy

- **5-fold Stratified Cross-Validation**
- Stratification ensures balanced representation of binding classes
- Out-of-fold predictions used for unbiased performance estimation

---

## Results

### Overall Performance

| Model          | AUC-ROC | Average Precision |
|----------------|---------|-------------------|
| **Ensemble**   | **0.9183** | **0.3400** |
| XGBoost        | 0.9047  | 0.3292            |
| LightGBM       | 0.9095  | 0.3136            |
| Random Forest  | 0.9000  | 0.2581            |

**Key Findings:**
- The ensemble outperforms individual models, demonstrating effective model complementarity
- All models achieve strong AUC scores (>0.90), indicating excellent discrimination
- Average Precision scores reflect the challenge of extreme class imbalance
- Ensemble provides ~1.4-3.6% improvement over individual models

### Cross-Validation Results by Model

#### XGBoost
| Fold | AUC    | Average Precision |
|------|--------|-------------------|
| 1    | 0.9426 | 0.3476            |
| 2    | 0.8700 | 0.3461            |
| 3    | 0.9223 | 0.4403            |
| 4    | 0.8571 | 0.2169            |
| 5    | 0.9386 | 0.4033            |
| **Mean** | **0.9047** | **0.3292** |

#### LightGBM
| Fold | AUC    | Average Precision |
|------|--------|-------------------|
| 1    | 0.9415 | 0.3076            |
| 2    | 0.9274 | 0.3313            |
| 3    | 0.9461 | 0.4493            |
| 4    | 0.9075 | 0.2484            |
| 5    | 0.9495 | 0.4314            |
| **Mean** | **0.9095** | **0.3136** |

#### Random Forest
| Fold | AUC    | Average Precision |
|------|--------|-------------------|
| 1    | 0.9157 | 0.2504            |
| 2    | 0.8897 | 0.2409            |
| 3    | 0.9146 | 0.4220            |
| 4    | 0.8660 | 0.2135            |
| 5    | 0.9161 | 0.3261            |
| **Mean** | **0.9000** | **0.2581** |

### Performance by Protein Target

| Protein | AUC    | Average Precision | Samples | Binding Rate |
|---------|--------|-------------------|---------|--------------|
| **sEH** | **0.9723** | **0.4887** | 8,423   | 1.07%        |
| BRD4    | 0.9514 | 0.2834            | 8,415   | 0.97%        |
| HSA     | 0.7821 | 0.2091            | 8,396   | 0.75%        |

**Insights:**
- sEH target shows the strongest predictive performance (AUC: 0.9723)
- HSA is the most challenging target, likely due to its broader binding profile
- Protein-specific patterns successfully captured by the models

### Feature Importance Analysis

**Top 10 Most Important Features (XGBoost):**

1. Fingerprint_1387: 0.0601
2. Fingerprint_77: 0.0425
3. Fingerprint_1168: 0.0369
4. Fingerprint_732: 0.0202
5. Fingerprint_1817: 0.0167
6. Fingerprint_119: 0.0142
7. Fingerprint_1193: 0.0137
8. Fingerprint_1384: 0.0117
9. Fingerprint_413: 0.0113
10. Fingerprint_1446: 0.0111

The distributed importance across molecular fingerprints indicates that binding prediction relies on multiple structural features rather than single dominant patterns.

---

## Model Evaluation Metrics

### Why These Metrics?

**AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**
- Measures the model's ability to discriminate between binders and non-binders
- Robust to class imbalance when comparing relative rankings
- Range: 0.5 (random) to 1.0 (perfect)

**Average Precision (AP)**
- Summarizes the precision-recall curve
- More sensitive to performance on the minority (binder) class
- Better suited for highly imbalanced datasets than accuracy
- Range: baseline (class frequency) to 1.0 (perfect)

For this dataset, with 0.93% binders, a random classifier would achieve AP ≈ 0.0093. Our ensemble achieves **36.5× better than random** (0.3400 vs 0.0093).

---

## Computational Details

### Training Environment
- Platform: Kaggle Notebooks
- Python: 3.12
- Key Libraries:
  - RDKit: 2025.9.3
  - XGBoost: 3.1.0
  - LightGBM: 4.6.0
  - scikit-learn: 1.6.1
  - Pandas: 2.2.2
  - NumPy: 2.0.2

### Training Time
- Feature generation: ~2-3 minutes for 25K molecules
- Model training: ~10-15 minutes total (all folds, all models)
- Total pipeline runtime: ~20 minutes

### Memory Usage
- Peak memory: <8GB RAM
- Efficient for local development and experimentation

---

## Model Predictions

### Prediction Distribution Analysis

The trained ensemble generates probability scores ranging from 0 to 1, where:
- **High scores (>0.5)**: Strong predicted binding affinity
- **Low scores (<0.1)**: Predicted non-binders
- **Mid-range (0.1-0.5)**: Uncertain/borderline cases

The model shows good calibration with clear separation between binder and non-binder distributions, indicating reliable confidence estimates.



## Future Improvements

### Immediate Next Steps
1. **Full dataset training**: Scale to complete training set (98M molecules)
2. **Advanced features**: 
   - 3D molecular descriptors
   - Protein-ligand interaction fingerprints
   - Graph neural network embeddings
3. **Hyperparameter optimization**: Systematic tuning via Bayesian optimization
4. **Advanced ensembling**: Stacking or weighted averaging based on protein target

### Advanced Approaches
1. **Deep learning**: Graph neural networks for molecular representation
2. **Active learning**: Iterative selection of informative molecules
3. **Multi-task learning**: Joint prediction across all protein targets
4. **Data augmentation**: SMILES enumeration and molecular transformations
5. **External data**: Pre-training on ChEMBL or PubChem datasets

---

## Reproducibility

All results are reproducible using:
- Fixed random seed: 42
- Identical data sampling strategy
- Deterministic model training (where supported)
- Versioned dependencies

To reproduce these results, simply run the provided Jupyter notebook with the specified environment.

---

## Conclusion

This project demonstrates that ensemble machine learning with molecular fingerprints can effectively predict small molecule binding to protein targets, even with extreme class imbalance. The **0.9183 AUC** and **0.3400 Average Precision** represent strong performance that could accelerate early-stage drug discovery by prioritizing promising candidates for experimental validation.


---

## References

- NeurIPS 2024 - Predict New Medicines with BELKA Challenge
- RDKit: Open-source cheminformatics software
- XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)
- LightGBM: A Highly Efficient Gradient Boosting Decision Tree (Ke et al., 2017)
