# 📦 Week 5 — Ensemble Classifier (Part 1)
> **Date:** 28 July | **Topic:** Random Forest, AdaBoost, Gradient Boosting, XGBoost, ExtraTrees, and Stacking

---

## 🎯 Learning Objectives

- Understand the intuition behind **Ensemble Methods** (combining weak learners into a strong learner).
- Learn the difference between **Bagging** (e.g., Random Forest) and **Boosting** (e.g., AdaBoost, Gradient Boosting).
- Implement various powerful ensemble algorithms: Random Forest, ExtraTrees, AdaBoost, Gradient Boosting, XGBoost, and Stacking.
- Evaluate the performance improvements gained from combining models.

---

## 📚 Key Concepts

| Concept | Description |
|---|---|
| **Ensemble Learning** | A machine learning paradigm where multiple models (often called "weak learners") are trained to solve the same problem and combined to get better results. |
| **Bagging (Bootstrap Aggregating)** | Involves training multiple models on different random subsets of the data (with replacement) and averaging their predictions (e.g., Random Forest). |
| **Boosting** | Involves training multiple models sequentially, where each new model focuses on correcting the errors made by the previous ones (e.g., AdaBoost, XGBoost). |
| **Stacking** | Involves training a "meta-model" to combine the predictions of several base models to make the final prediction. |

---

## 🛠️ Quick Setup

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensemble Models
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
```

---

## 📁 Files in This Week

| File | Description |
|---|---|
| `lab-week05/RandomForest.ipynb` | Implementation and tuning of Random Forest |
| `lab-week05/AdaBoost_Classifier.ipynb` | Implementation of AdaBoost |
| `lab-week05/Gradient_Boosting_&_XGB_Classifier.ipynb` | Implementation of Gradient Boosting and XGBoost |
| `lab-week05/ExtraTrees.ipynb` | Implementation of Extra Trees Classifier |
| `lab-week05/StackingCVClassifier.ipynb` | Implementation of Stacking using Cross-Validation |
| `slides/06 Ensemble Classifier.pdf` | Lecture slides covering Ensemble Methods |

---

*Applied Machine Learning — DSBA8 | Week 5*
