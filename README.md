# Hotel Booking Cancellation Prediction

A COMP90049 team project comparing hotel-booking cancellation models, feature ablation and cross-hotel evaluation. Yiqun Qiao contributed decision-tree and neural-network modelling and evaluation.

## Source

Copied from the team repository [YuxuanLin050309/IML-Assignment2](https://github.com/YuxuanLin050309/IML-Assignment2). This is a separate private source snapshot; original notebook attribution is retained.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `preprocessing.ipynb` | Data preparation and shared train/validation/test files |
| `logistic_regression.ipynb` | Logistic-regression experiments |
| `naive_bayes.ipynb` | Naive Bayes experiments |
| `decision_tree.ipynb` | Decision-tree experiments |
| `neural_network.ipynb` | Neural-network experiments |

## Preparation and execution

The notebooks depend on the original hotel-booking data and generated shared data files. Datasets, cached splits, plots, reports and IDE settings are not included. Place `hotel_bookings.csv` at the repository root and configure the original environment. Run `preprocessing.ipynb` to generate `shared_data/train_clean.csv`, `val_clean.csv`, `test_clean.csv` and `split_indices.npz`. Then run `logistic_regression.ipynb` before `naive_bayes.ipynb`, whose comparison cells read `outputs/logistic_regression_error_analysis.csv`. The decision-tree and neural-network notebooks use the same shared data splits. Python dependencies are listed in `requirements.txt`.

The preprocessing notebook retains an `import features` statement from the source repository, which does not contain that module. The original environment must supply it, or the import needs review before execution. This upload does not modify that code.

Notebook code and Markdown cells are retained. Saved execution outputs and execution counts are cleared in the upload copies. Source syntax and notebook structure were checked; model training was not rerun during this upload.
