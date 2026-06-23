# Statistical Machine Learning — Assignments

Numpy-only implementations of core ML algorithms, built from scratch as part of a Statistical Machine Learning course. No sklearn for the core math — only numpy + matplotlib. Each folder = one assignment, with code + PDF report.

## Contents

| Folder | Topics | Dataset |
|---|---|---|
| `A1_Bayes_Classifiers_LDA_QDA` | LDA, QDA, MLE for Gaussian class params, t-SNE visualization | MNIST (digits 0,1,2) |
| `A2_PCA_FDA_Dimensionality_Reduction` | PCA (eigen-decomp from scratch), Fisher Discriminant Analysis, image reconstruction | MNIST (digits 0,1,2) |
| `A3_Decision_Trees_Bagging_RandomForest` | Decision Trees, Bagging, Random Forest, Ridge/Lasso Regression | MNIST / Fashion-MNIST |
| `A4_Boosting_Perceptron` | AdaBoost, Gradient Boosting (regression stumps), Rosenblatt Perceptron | MNIST (digits 4 vs 9), synthetic 2D data |

## How to run

Each `.py` file is standalone — it downloads its dataset automatically (or expects the relevant `.npz` next to it) and produces console output + matplotlib plots. Requires `numpy`, `matplotlib`, and `scikit-learn` (only used for t-SNE in A1).

```bash
pip install numpy matplotlib scikit-learn
python A1_Bayes_Classifiers_LDA_QDA/A1_lda_qda_bayes.py
```

## Notes

- All datasets (MNIST/Fashion-MNIST) are auto-downloaded on first run and gitignored — don't need to be committed.
- Each folder's report PDF has methodology, plots, and result discussion.
