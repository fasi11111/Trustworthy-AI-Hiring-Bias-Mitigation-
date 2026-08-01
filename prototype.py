"""
IS4T702 MSc Project — Basic Prototype
=======================================
Trustworthy AI: A Comparative Experimental Study of Bias Mitigation
Techniques in AI Driven Hiring Algorithms

Student  : Faisal Aleem
Supervisor: Janusz
University: University of South Wales
Year      : 2026

OVERVIEW
--------
This prototype implements the core experimental pipeline described in the
project proposal. It loads the UCI Adult Income dataset, trains a baseline
logistic regression classifier, applies four bias mitigation techniques
spanning all three families (preprocessing, in processing, postprocessing),
computes standard fairness metrics for each approach, prints a comparative
results table, and saves the results to CSV and PNG files.

INSTALL DEPENDENCIES
---------------------
    pip install aif360 scikit-learn pandas numpy matplotlib

AIF360 bundles the UCI Adult dataset internally, so no separate download
is required. If the dataset is missing, AIF360 prints instructions on
where to place the raw files.

USAGE
-----
    python prototype.py

OUTPUTS
-------
    bias_mitigation_results.csv   Comparative fairness metrics table
    bias_mitigation_chart.png     Bar chart comparing approaches
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing

try:
    from aif360.algorithms.inprocessing import PrejudiceRemover
    PREJUDICE_REMOVER_AVAILABLE = True
except ImportError:
    PREJUDICE_REMOVER_AVAILABLE = False


# ================================================================
# CONFIGURATION
# ================================================================

PROTECTED_ATTRIBUTE = 'sex'
PRIVILEGED_GROUPS   = [{'sex': 1}]   # 1 = Male
UNPRIVILEGED_GROUPS = [{'sex': 0}]   # 0 = Female
RANDOM_STATE        = 42
TRAIN_RATIO         = 0.7


# ================================================================
# SECTION 1: DATA LOADING
# ================================================================

def load_dataset():
    """
    Load the UCI Adult Income dataset via AIF360.

    The dataset predicts whether an individual earns more than $50,000
    per year based on census attributes. Protected attribute: sex.

    Reference:
        Dua, D. and Graff, C. (2019) UCI Machine Learning Repository.
        Irvine, CA: University of California.

    Returns:
        train_data : BinaryLabelDataset (training split)
        test_data  : BinaryLabelDataset (test split)
    """
    print("\n" + "=" * 65)
    print("  SECTION 1: Loading UCI Adult Income Dataset")
    print("=" * 65)

    dataset = AdultDataset(
        protected_attribute_names=[PROTECTED_ATTRIBUTE],
        privileged_classes=[lambda x: x > 0],
        features_to_drop=['race']
    )

    train_data, test_data = dataset.split(
        [TRAIN_RATIO], shuffle=True, seed=RANDOM_STATE
    )

    print(f"  Protected attribute : {PROTECTED_ATTRIBUTE} (1=Male, 0=Female)")
    print(f"  Training samples    : {len(train_data.features)}")
    print(f"  Test samples        : {len(test_data.features)}")
    print(f"  Feature count       : {train_data.features.shape[1]}")

    return train_data, test_data


# ================================================================
# SECTION 2: BASELINE CLASSIFIER
# ================================================================

def baseline(train_data, test_data):
    """
    Train a logistic regression classifier with no fairness intervention.

    This provides the reference point for evaluating how much each
    mitigation technique improves fairness and at what cost to accuracy.

    Reference:
        Pedregosa, F. et al. (2011) Scikit-learn: Machine learning in
        Python. Journal of Machine Learning Research, 12, pp. 2825-2830.

    Returns:
        test_pred : BinaryLabelDataset with predicted labels
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_data.features)
    y_train = train_data.labels.ravel()
    X_test  = scaler.transform(test_data.features)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    clf.fit(X_train, y_train)

    test_pred        = test_data.copy()
    test_pred.labels = clf.predict(X_test).reshape(-1, 1)

    return test_pred


# ================================================================
# SECTION 3: PREPROCESSING — REWEIGHING
# ================================================================

def apply_reweighing(train_data, test_data):
    """
    Apply the Reweighing preprocessing technique.

    This technique assigns instance weights to the training data so
    that the weighted distribution is independent of the protected
    attribute. Instances representing historically underrepresented
    outcomes for a group receive higher weights.

    Reference:
        Kamiran, F. and Calders, T. (2012) Data preprocessing techniques
        for classification without discrimination. Knowledge and
        Information Systems, 33(1), pp. 1-33.

    Returns:
        test_pred : BinaryLabelDataset with predicted labels
    """
    rw = Reweighing(
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        privileged_groups=PRIVILEGED_GROUPS
    )
    train_rw = rw.fit_transform(train_data)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_rw.features)
    y_train = train_rw.labels.ravel()
    weights = train_rw.instance_weights
    X_test  = scaler.transform(test_data.features)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    clf.fit(X_train, y_train, sample_weight=weights)

    test_pred        = test_data.copy()
    test_pred.labels = clf.predict(X_test).reshape(-1, 1)

    return test_pred


# ================================================================
# SECTION 4: PREPROCESSING — DISPARATE IMPACT REMOVER
# ================================================================

def apply_dir(train_data, test_data):
    """
    Apply the Disparate Impact Remover preprocessing technique.

    This technique repairs feature values to reduce correlation
    with the protected attribute, based on the four-fifths rule
    used in employment discrimination analysis. The repair_level
    parameter (0.0 to 1.0) controls the degree of repair applied.

    Reference:
        Feldman, M. et al. (2015) Certifying and removing disparate
        impact. Proceedings of the 21st ACM SIGKDD International
        Conference on Knowledge Discovery and Data Mining, pp. 259-268.

    Returns:
        test_pred : BinaryLabelDataset with predicted labels
    """
    dir_transform = DisparateImpactRemover(
        repair_level=0.8,
        sensitive_attribute=PROTECTED_ATTRIBUTE
    )

    train_dir = dir_transform.fit_transform(train_data)
    test_dir  = dir_transform.fit_transform(test_data)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_dir.features)
    y_train = train_dir.labels.ravel()
    X_test  = scaler.transform(test_dir.features)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    clf.fit(X_train, y_train)

    test_pred        = test_data.copy()
    test_pred.labels = clf.predict(X_test).reshape(-1, 1)

    return test_pred


# ================================================================
# SECTION 5: IN-PROCESSING — PREJUDICE REMOVER
# ================================================================

def apply_prejudice_remover(train_data, test_data):
    """
    Apply the Prejudice Remover in-processing technique.

    This technique adds a fairness-aware regularisation term to the
    logistic regression learning objective. The eta parameter controls
    the strength of the regularisation; higher values enforce greater
    fairness at the cost of accuracy.

    Reference:
        Kamishima, T. et al. (2012) Fairness-aware classifier with
        prejudice remover regularizer. ECML PKDD 2012, LNCS 7524,
        pp. 35-50.

    Returns:
        test_pred  : BinaryLabelDataset with predicted labels, or
        None       : if PrejudiceRemover is unavailable
    """
    if not PREJUDICE_REMOVER_AVAILABLE:
        print("  PrejudiceRemover not available — skipping.")
        return None

    pr = PrejudiceRemover(
        eta=25.0,
        sensitive_attr=PROTECTED_ATTRIBUTE,
        class_attr=train_data.label_names[0]
    )
    pr.fit(train_data)
    test_pred = pr.predict(test_data)

    return test_pred


# ================================================================
# SECTION 6: POSTPROCESSING — EQUALISED ODDS
# ================================================================

def apply_eq_odds(train_data, test_data):
    """
    Apply the Equalised Odds postprocessing technique.

    This technique adjusts classifier decision thresholds per group
    to achieve equal true positive rates and equal false positive
    rates across protected groups. It operates on the classifier
    outputs without requiring retraining.

    Reference:
        Hardt, M., Price, E. and Srebro, N. (2016) Equality of
        opportunity in supervised learning. Advances in Neural
        Information Processing Systems 29 (NeurIPS 2016).

    Returns:
        test_pred_eq : BinaryLabelDataset with adjusted labels
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_data.features)
    X_test  = scaler.transform(test_data.features)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)
    clf.fit(X_train, train_data.labels.ravel())

    train_pred        = train_data.copy()
    train_pred.scores = clf.predict_proba(X_train)[:, 1].reshape(-1, 1)
    train_pred.labels = clf.predict(X_train).reshape(-1, 1)

    test_pred_prob        = test_data.copy()
    test_pred_prob.scores = clf.predict_proba(X_test)[:, 1].reshape(-1, 1)
    test_pred_prob.labels = clf.predict(X_test).reshape(-1, 1)

    eq = EqOddsPostprocessing(
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        privileged_groups=PRIVILEGED_GROUPS,
        seed=RANDOM_STATE
    )
    eq.fit(train_data, train_pred)
    test_pred_eq = eq.predict(test_pred_prob)

    return test_pred_eq


# ================================================================
# SECTION 7: FAIRNESS METRIC EVALUATION
# ================================================================

def evaluate(test_data, predicted, label):
    """
    Compute accuracy and four standard fairness metrics.

    Metrics computed:
    - Accuracy            : overall classification accuracy
    - Dem. Parity Diff.   : statistical parity difference (0 = fair)
    - Avg. Odds Diff.     : average of TPR and FPR differences (0 = fair)
    - Equal Odds Diff.    : max of TPR and FPR differences (0 = fair)
    - Disparate Impact    : ratio of positive outcome rates (1 = fair)

    References:
        Hardt et al. (2016); Feldman et al. (2015);
        Bellamy et al. (2019) AIF360.

    Returns:
        dict : metric name -> value
    """
    cm = ClassificationMetric(
        test_data, predicted,
        unprivileged_groups=UNPRIVILEGED_GROUPS,
        privileged_groups=PRIVILEGED_GROUPS
    )

    accuracy = np.mean(predicted.labels == test_data.labels)

    return {
        'Approach'           : label,
        'Accuracy'           : round(float(accuracy), 4),
        'Dem. Parity Diff.'  : round(float(cm.statistical_parity_difference()), 4),
        'Avg. Odds Diff.'    : round(float(cm.average_odds_difference()), 4),
        'Equal Odds Diff.'   : round(float(cm.equalized_odds_difference()), 4),
        'Disparate Impact'   : round(float(cm.disparate_impact()), 4),
    }


# ================================================================
# SECTION 8: VISUALISATION
# ================================================================

def visualise(df):
    """
    Produce a bar chart comparing accuracy and fairness metrics
    across all approaches. Saves the chart to PNG.

    Returns:
        None
    """
    approaches = df['Approach'].tolist()
    colours    = ['#5F5E5A', '#378ADD', '#0F6E56', '#534AB7', '#BA7517']
    n          = len(approaches)

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        'Bias Mitigation Comparison — UCI Adult Income Dataset\n'
        'Protected Attribute: Sex (Male vs. Female)   |   '
        'IS4T702 MSc Project · Faisal Aleem',
        fontsize=11, fontweight='bold', y=1.01
    )

    metrics = [
        ('Accuracy',          'Accuracy (higher = better)',  True,  (0.7, 0.9)),
        ('Dem. Parity Diff.', 'Dem. Parity Diff. (|x|, lower = fairer)', False, None),
        ('Avg. Odds Diff.',   'Avg. Odds Diff. (|x|, lower = fairer)',   False, None),
        ('Disparate Impact',  'Disparate Impact (closer to 1 = fairer)', True,  None),
    ]

    for ax, (col, title, raw, ylim) in zip(axes, metrics):
        values = df[col].abs().tolist() if not raw else df[col].tolist()
        bars   = ax.bar(range(n), values, color=colours[:n], width=0.55)
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(approaches, rotation=30, ha='right', fontsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8
            )
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.savefig('bias_mitigation_chart.png', dpi=150, bbox_inches='tight')
    print("\n  Chart saved to: bias_mitigation_chart.png")


# ================================================================
# MAIN PIPELINE
# ================================================================

def main():
    print("\n" + "=" * 65)
    print("  IS4T702 MSc Project — Bias Mitigation Prototype")
    print("  Student   : Faisal Aleem")
    print("  Supervisor: Janusz")
    print("  Dataset   : UCI Adult Income (protected attribute: sex)")
    print("=" * 65)

    # --- Load data ---
    train_data, test_data = load_dataset()

    # --- Run all approaches ---
    print("\n" + "=" * 65)
    print("  SECTION 2-6: Training Classifiers and Applying Mitigation")
    print("=" * 65)

    results = []

    print("\n  [1/5] Baseline (no mitigation) ...")
    baseline_pred = baseline(train_data, test_data)
    results.append(evaluate(test_data, baseline_pred, 'Baseline'))
    print("        Done.")

    print("\n  [2/5] Preprocessing: Reweighing ...")
    rw_pred = apply_reweighing(train_data, test_data)
    results.append(evaluate(test_data, rw_pred, 'Reweighing'))
    print("        Done.")

    print("\n  [3/5] Preprocessing: Disparate Impact Remover ...")
    dir_pred = apply_dir(train_data, test_data)
    results.append(evaluate(test_data, dir_pred, 'Dispar. Impact Rem.'))
    print("        Done.")

    print("\n  [4/5] In-processing: Prejudice Remover ...")
    pr_pred = apply_prejudice_remover(train_data, test_data)
    if pr_pred is not None:
        results.append(evaluate(test_data, pr_pred, 'Prejudice Remover'))
        print("        Done.")

    print("\n  [5/5] Postprocessing: Equalised Odds ...")
    eq_pred = apply_eq_odds(train_data, test_data)
    results.append(evaluate(test_data, eq_pred, 'Equalised Odds'))
    print("        Done.")

    # --- Display results ---
    df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("  SECTION 7: Comparative Fairness Metrics")
    print("=" * 65)
    print(df.to_string(index=False))
    print("\n  Notes:")
    print("  - Dem. Parity Diff.  : 0 = perfect demographic parity")
    print("  - Avg. Odds Diff.    : 0 = perfect average odds")
    print("  - Equal Odds Diff.   : 0 = perfect equalised odds")
    print("  - Disparate Impact   : 1 = perfect parity (< 0.8 = adverse)")
    print("=" * 65)

    # --- Save results ---
    df.to_csv('bias_mitigation_results.csv', index=False)
    print("\n  Results saved to: bias_mitigation_results.csv")

    # --- Visualise ---
    print("\n" + "=" * 65)
    print("  SECTION 8: Generating Visualisation")
    print("=" * 65)
    visualise(df)

    print("\n" + "=" * 65)
    print("  Prototype pipeline complete.")
    print("  Next steps: extend to COMPAS dataset, add more classifiers,")
    print("  implement in-processing adversarial debiasing, and run")
    print("  statistical significance tests across multiple trials.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
