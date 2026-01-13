# NeurIPS 2024 BELKA: Predicting Small Molecule Binding

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)](https://www.kaggle.com/competitions/leash-BELKA)

> **Ensemble machine learning pipeline for predicting small molecule binding affinity to protein targets**  
> *Achieving 0.9183 AUC-ROC through XGBoost, LightGBM, and Random Forest ensemble*

---

## Project Overview

This repository contains a complete machine learning pipeline for the [NeurIPS 2024 - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA) competition. The goal is to predict whether small molecules will bind to specific protein targets—a critical step in drug discovery.

### Key Features
-  **Morgan fingerprint** molecular representation using RDKit
-  **Ensemble learning** with XGBoost, LightGBM, and Random Forest
-  **5-fold stratified cross-validation** for robust evaluation
-  **Strong performance**: 0.9183 AUC-ROC, 0.3400 Average Precision
-  **Efficient**: Trains in ~20 minutes on standard hardware

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/neurips-2024-belka.git
cd neurips-2024-belka

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook neurips-2024-predict-new-medicines-with-belka.ipynb
```

---

##  Dataset

### Overview
- **Source**: Leash Biosciences BELKA dataset
- **Training size**: 98M molecule-protein pairs (sampled 25K for efficiency)
- **Test size**: 1.67M molecules
- **Protein targets**: 3 (BRD4, HSA, sEH)
- **Format**: SMILES strings + binding labels

### Target Distribution

| Protein | Samples | Binding Rate |
|---------|---------|--------------|
| BRD4    | 8,415   | 0.97%        |
| HSA     | 8,396   | 0.75%        |
| sEH     | 8,423   | 1.07%        |
| **Total** | **25,234** | **0.93%** |

The dataset exhibits extreme class imbalance (~99% non-binders), reflecting real-world drug discovery challenges.

---

##  Methodology

### Pipeline Architecture

```
Raw SMILES → Morgan Fingerprints → Feature Matrix → Ensemble Models → Predictions
                (RDKit)              (+Protein)      (XGB+LGB+RF)
```

### 1. Feature Engineering

**Molecular Representation**
- **Morgan Fingerprints**: 2048-bit circular fingerprints (radius=2)
- Captures structural and chemical properties
- Generated using RDKit's `MorganGenerator`

**Protein Encoding**
- One-hot encoding for 3 protein targets
- Enables learning of protein-specific binding patterns

**Final Features**: 2,051 dimensions

### 2. Model Training

**Three complementary models:**

| Model | Key Parameters |
|-------|---------------|
| **XGBoost** | max_depth=8, lr=0.05, n_est=300 |
| **LightGBM** | num_leaves=64, lr=0.05, n_est=300 |
| **Random Forest** | n_est=200, max_depth=20 |

**Ensemble Strategy**: Simple averaging of predicted probabilities

### 3. Validation

- **5-fold Stratified Cross-Validation**
- Out-of-fold predictions for unbiased evaluation
- Metrics: AUC-ROC (discrimination) + Average Precision (imbalanced data)

---

##  Results

### Overall Performance

| Model | AUC-ROC | Average Precision |
|-------|---------|-------------------|
| **Ensemble** | **0.9183** | **0.3400** |
| XGBoost | 0.9047 | 0.3292 |
| LightGBM | 0.9095 | 0.3136 |
| Random Forest | 0.9000 | 0.2581 |

### Performance by Protein Target

| Protein | AUC-ROC | AP | Difficulty |
|---------|---------|-----|-----------|
| sEH | 0.9723  | 0.4887 | Easiest |
| BRD4 | 0.9514 | 0.2834 | Moderate |
| HSA | 0.7821 | 0.2091 | Hardest |

### Key Insights

 **Ensemble improves over individual models** by 1.4-3.6%  
 **Strong discrimination** across all protein targets (AUC > 0.78)  
 **36.5× better than random** on minority class (AP: 0.34 vs baseline 0.0093)  
 **Robust across folds** (CV std < 0.04 for AUC)



##  Installation

### Requirements
- Python 3.12+
- 8GB+ RAM recommended
- CUDA optional (for GPU acceleration)

### Dependencies

```bash
pip install rdkit pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Environment Setup

**Option 1: Conda**
```bash
conda create -n belka python=3.12
conda activate belka
pip install -r requirements.txt
```

**Option 2: venv**
```bash
python -m venv belka-env
source belka-env/bin/activate  # On Windows: belka-env\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### Running the Pipeline

**1. Complete Pipeline (Recommended)**
```python
# Open and run the Jupyter notebook
jupyter notebook neurips-2024-predict-new-medicines-with-belka.ipynb
```

**2. Training Models**
```python
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Load your data
X, y = load_features_and_labels()

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train XGBoost
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model = xgb.XGBClassifier(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        early_stopping_rounds=30
    )
    model.fit(X[train_idx], y[train_idx], 
              eval_set=[(X[val_idx], y[val_idx])])
```

**3. Making Predictions**
```python
import joblib

# Load trained models
xgb_models = joblib.load('models/xgboost_models.pkl')
lgb_models = joblib.load('models/lightgbm_models.pkl')
rf_models = joblib.load('models/randomforest_models.pkl')

# Average predictions across folds
xgb_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in xgb_models], axis=0)
lgb_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in lgb_models], axis=0)
rf_preds = np.mean([m.predict_proba(X_test)[:, 1] for m in rf_models], axis=0)

# Ensemble
ensemble_preds = (xgb_preds + lgb_preds + rf_preds) / 3
```

---

##  Model Details

### XGBoost Configuration

```python
XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    max_depth=8,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
    early_stopping_rounds=30
)
```

### LightGBM Configuration

```python
LGBMClassifier(
    objective='binary',
    metric='auc',
    num_leaves=64,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

---

## Performance Analysis

### Cross-Validation Stability

**XGBoost Fold Performance:**
- Best fold: 0.9426 AUC (Fold 1)
- Worst fold: 0.8571 AUC (Fold 4)
- Standard deviation: 0.036

The models show good stability across folds, indicating robust performance.

### Feature Importance

**Top contributing features:**
- Molecular fingerprints dominate importance (bits 1387, 77, 1168)
- Distributed importance across multiple bits
- No single dominant feature (max importance: 6.01%)

### Error Analysis

**Model strengths:**
- Excellent at identifying clear non-binders (high specificity)
- Good discrimination of strong binders (high AUC)

**Model limitations:**
- Moderate precision on positive class (AP ~0.34)
- Some false positives due to extreme imbalance
- HSA target more challenging than others

---

## Future Work

### Short-term Improvements
- [ ] Train on full 98M molecule dataset
- [ ] Hyperparameter optimization (Optuna/Bayesian)
- [ ] Weighted ensemble based on protein target
- [ ] Additional molecular descriptors (MACCS keys, RDKit descriptors)

### Medium-term Enhancements
- [ ] Graph Neural Networks (GNN) for molecular representation
- [ ] Attention-based models for protein-molecule interactions
- [ ] Multi-task learning across all protein targets
- [ ] Data augmentation via SMILES enumeration

### Long-term Research
- [ ] 3D conformer-based features
- [ ] Transfer learning from ChEMBL/PubChem
- [ ] Active learning for efficient labeling
- [ ] Explainable AI for binding site identification

---


##  References

### Papers
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NIPS 2017*.
- Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*.

### Resources
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Competition Page](https://www.kaggle.com/competitions/leash-BELKA)

---

