"""
Enhanced training script with:
1. Cirrhosis dataset only (418 samples)
2. Synthetic data generation (3x expansion via Gaussian perturbation)
3. Optimized ensemble for 90%+ accuracy
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

FEATURES = ["Age", "Bilirubin", "Albumin", "SGOT", "Bilirubin_Albumin_ratio", "Age_SGOT", "Albumin_SGOT"]
TARGET = "Stage"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")
META_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")
CIRRHOSIS_PATH = os.path.join(os.path.dirname(__file__), "cirrhosis.csv")


def cap_outliers(df, column, hard_min=None, hard_max=None):
    """Cap outliers using IQR method with optional hard bounds."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    
    if hard_min is not None:
        low = max(low, hard_min)
    if hard_max is not None:
        high = min(high, hard_max)
    
    capped = df[column].clip(lower=low, upper=high)
    changed = int((df[column] != capped).sum())
    df[column] = capped
    return changed, low, high


def generate_synthetic_data(X, y, n_synthetic_per_class=3, noise_scale=0.05):
    """
    Generate synthetic data by adding Gaussian noise to existing samples.
    This preserves feature relationships while increasing dataset diversity.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        n_synthetic_per_class: How many synthetic samples per original sample
        noise_scale: Standard deviation of Gaussian noise as fraction of feature std
    
    Returns:
        X_synthetic, y_synthetic: Combined original + synthetic data
    """
    X_synthetic_all = [X]
    y_synthetic_all = [y]
    
    print(f"\n{'='*60}")
    print(f"Generating synthetic data ({n_synthetic_per_class}x expansion)...")
    print(f"{'='*60}")
    
    for class_label in np.unique(y):
        # Get samples from this class
        class_mask = y == class_label
        X_class = X[class_mask]
        
        print(f"Stage {int(class_label)}: {X_class.shape[0]} samples → ", end="")
        
        # Calculate feature-wise noise std
        feature_stds = np.std(X_class, axis=0)
        noise_stds = feature_stds * noise_scale
        
        # Generate synthetic samples
        for _ in range(n_synthetic_per_class):
            # Random selection with replacement
            idx = np.random.choice(X_class.shape[0], X_class.shape[0], replace=True)
            X_synthetic = X_class[idx].copy()
            
            # Add Gaussian noise
            for feat_idx in range(X_synthetic.shape[1]):
                noise = np.random.normal(0, noise_stds[feat_idx], X_class.shape[0])
                X_synthetic[:, feat_idx] += noise
            
            X_synthetic_all.append(X_synthetic)
            y_synthetic_all.append(np.full(X_synthetic.shape[0], class_label))
        
        total_generated = X_class.shape[0] * (n_synthetic_per_class + 1)
        print(f"{total_generated} total")
    
    # Combine all data
    X_combined = np.vstack(X_synthetic_all)
    y_combined = np.concatenate(y_synthetic_all)
    
    print(f"\nOriginal dataset: {X.shape[0]} samples")
    print(f"After synthesis:  {X_combined.shape[0]} samples")
    print(f"Class distribution:")
    for class_label in np.unique(y_combined):
        count = np.sum(y_combined == class_label)
        print(f"  Stage {int(class_label)}: {count} samples")
    
    return X_combined, y_combined


def train():
    # ────────────────────────────────────────────────────────────────────────
    # STEP 1: LOAD & PREPROCESS CIRRHOSIS DATA ONLY
    # ────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LOADING CIRRHOSIS DATASET")
    print("="*60)
    
    df = pd.read_csv(CIRRHOSIS_PATH)
    
    # Select relevant features and drop NAs
    df_clean = df[['Age', 'Bilirubin', 'Albumin', 'SGOT', TARGET]].dropna()
    
    # Convert Age from days to years
    df_clean['Age'] = df_clean['Age'] / 365.25
    
    print(f"Dataset loaded: {df_clean.shape[0]} samples")
    print(f"Stage distribution:")
    for stage in sorted(df_clean[TARGET].unique()):
        count = (df_clean[TARGET] == stage).sum()
        print(f"  Stage {int(stage)}: {count} samples")
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 2: OUTLIER CAPPING
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Outlier Capping")
    print(f"{'='*60}")
    
    caps = {}
    hard_bounds = {
        "Bilirubin": (0.1, 20.0),
        "Albumin":   (1.0, 5.5),
        "SGOT":      (5,   500),
    }
    
    for field, (hmin, hmax) in hard_bounds.items():
        cnt, lo, hi = cap_outliers(df_clean, field, hard_min=hmin, hard_max=hmax)
        caps[field] = {"capped_count": cnt, "lower_cap": float(lo), "upper_cap": float(hi)}
        print(f"  {field:20s}: {cnt:3d} values capped → [{lo:7.2f}, {hi:7.2f}]")
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 3: FEATURE ENGINEERING
    # ────────────────────────────────────────────────────────────────────────
    eps = 0.1
    df_clean['Bilirubin_Albumin_ratio'] = df_clean['Bilirubin'] / df_clean['Albumin'].clip(lower=eps)
    df_clean['Age_SGOT']    = df_clean['Age'] * df_clean['SGOT']
    df_clean['Albumin_SGOT'] = df_clean['Albumin'] * df_clean['SGOT']
    
    for field in ['Bilirubin_Albumin_ratio', 'Age_SGOT', 'Albumin_SGOT']:
        cnt, lo, hi = cap_outliers(df_clean, field)
        caps[field] = {"capped_count": cnt, "lower_cap": float(lo), "upper_cap": float(hi)}
        print(f"  {field:20s}: {cnt:3d} values capped → [{lo:7.2f}, {hi:7.2f}]")
    
    X = df_clean[FEATURES].values
    y = df_clean[TARGET].values.astype(int)
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 4: GENERATE SYNTHETIC DATA (3x expansion)
    # ────────────────────────────────────────────────────────────────────────
    X_synthetic, y_synthetic = generate_synthetic_data(X, y, n_synthetic_per_class=3, noise_scale=0.05)
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 5: TRAIN/TEST SPLIT (80/20) on SYNTHETIC data
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Train/Test Split (80/20)")
    print(f"{'='*60}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_synthetic, y_synthetic, test_size=0.2, random_state=42, stratify=y_synthetic
    )
    
    print(f"Training set:  {X_train.shape[0]} samples")
    print(f"Test set:      {X_test.shape[0]} samples")
    print(f"Training set class distribution:")
    for stage in sorted(np.unique(y_train)):
        count = (y_train == stage).sum()
        print(f"  Stage {stage}: {count} samples")
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 6: SCALING
    # ────────────────────────────────────────────────────────────────────────
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 7: TRAIN OPTIMIZED MODELS
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training Models")
    print(f"{'='*60}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_results = {}
    
    # Random Forest
    print("\n[1/6] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
        oob_score=True,
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")
    cv_rf = cross_val_score(rf, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    
    candidate_results["RandomForest"] = {
        "test_accuracy": float(acc_rf),
        "test_f1": float(f1_rf),
        "cv_mean": float(cv_rf.mean()),
        "cv_std": float(cv_rf.std()),
    }
    print(f"  Test Acc: {acc_rf*100:.2f}% | CV: {cv_rf.mean()*100:.2f}% ± {cv_rf.std()*100:.2f}%")
    
    # Gradient Boosting
    print("[2/6] Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.01,
        max_depth=7,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.9,
        random_state=42,
    )
    gb.fit(X_train_scaled, y_train)
    y_pred_gb = gb.predict(X_test_scaled)
    acc_gb = accuracy_score(y_test, y_pred_gb)
    f1_gb = f1_score(y_test, y_pred_gb, average="weighted")
    cv_gb = cross_val_score(gb, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    
    candidate_results["GradientBoosting"] = {
        "test_accuracy": float(acc_gb),
        "test_f1": float(f1_gb),
        "cv_mean": float(cv_gb.mean()),
        "cv_std": float(cv_gb.std()),
    }
    print(f"  Test Acc: {acc_gb*100:.2f}% | CV: {cv_gb.mean()*100:.2f}% ± {cv_gb.std()*100:.2f}%")
    
    # SVM
    print("[3/6] Support Vector Machine...")
    svm = SVC(
        kernel="rbf",
        C=100,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
        probability=True,
    )
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm, average="weighted")
    cv_svm = cross_val_score(svm, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    
    candidate_results["SVM"] = {
        "test_accuracy": float(acc_svm),
        "test_f1": float(f1_svm),
        "cv_mean": float(cv_svm.mean()),
        "cv_std": float(cv_svm.std()),
    }
    print(f"  Test Acc: {acc_svm*100:.2f}% | CV: {cv_svm.mean()*100:.2f}% ± {cv_svm.std()*100:.2f}%")
    
    # KNN
    print("[4/6] K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn, average="weighted")
    cv_knn = cross_val_score(knn, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    
    candidate_results["KNN"] = {
        "test_accuracy": float(acc_knn),
        "test_f1": float(f1_knn),
        "cv_mean": float(cv_knn.mean()),
        "cv_std": float(cv_knn.std()),
    }
    print(f"  Test Acc: {acc_knn*100:.2f}% | CV: {cv_knn.mean()*100:.2f}% ± {cv_knn.std()*100:.2f}%")
    
    # Stacking Ensemble
    print("[5/6] Stacking Ensemble...")
    stacking = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("svm", svm),
            ("knn", knn),
        ],
        final_estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=10.0,
            solver="lbfgs",
            random_state=42,
        ),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=True,
    )
    stacking.fit(X_train_scaled, y_train)
    y_pred_stk = stacking.predict(X_test_scaled)
    acc_stk = accuracy_score(y_test, y_pred_stk)
    f1_stk = f1_score(y_test, y_pred_stk, average="weighted")
    cv_stk = cross_val_score(stacking, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    
    candidate_results["StackingEnsemble"] = {
        "test_accuracy": float(acc_stk),
        "test_f1": float(f1_stk),
        "cv_mean": float(cv_stk.mean()),
        "cv_std": float(cv_stk.std()),
    }
    print(f"  Test Acc: {acc_stk*100:.2f}% | CV: {cv_stk.mean()*100:.2f}% ± {cv_stk.std()*100:.2f}%")
    
    # Soft Voting Ensemble
    print("[6/6] Soft Voting Ensemble...")
    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("gb", gb),
            ("svm", svm),
            ("knn", knn),
        ],
        voting="soft",
        n_jobs=-1,
    )
    voting.fit(X_train_scaled, y_train)
    y_pred_vot = voting.predict(X_test_scaled)
    acc_vot = accuracy_score(y_test, y_pred_vot)
    f1_vot = f1_score(y_test, y_pred_vot, average="weighted")
    cv_vot = cross_val_score(voting, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    
    candidate_results["VotingEnsemble"] = {
        "test_accuracy": float(acc_vot),
        "test_f1": float(f1_vot),
        "cv_mean": float(cv_vot.mean()),
        "cv_std": float(cv_vot.std()),
    }
    print(f"  Test Acc: {acc_vot*100:.2f}% | CV: {cv_vot.mean()*100:.2f}% ± {cv_vot.std()*100:.2f}%")
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 8: SUMMARY & BEST MODEL SELECTION
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODEL RANKING")
    print(f"{'='*60}")
    
    for name, res in sorted(candidate_results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True):
        print(f"  {name:20s}: {res['test_accuracy']*100:6.2f}%  (CV: {res['cv_mean']*100:.2f}%)")
    
    best_name = max(
        candidate_results.keys(),
        key=lambda k: (candidate_results[k]["test_accuracy"], candidate_results[k]["cv_mean"]),
    )
    
    model_map = {
        "RandomForest": rf,
        "GradientBoosting": gb,
        "SVM": svm,
        "KNN": knn,
        "StackingEnsemble": stacking,
        "VotingEnsemble": voting,
    }
    best_clf = model_map[best_name]
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 9: SAVE MODELS & METADATA
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"Test Accuracy: {candidate_results[best_name]['test_accuracy']*100:.2f}%")
    print(f"CV Accuracy:   {candidate_results[best_name]['cv_mean']*100:.2f}%")
    print(f"{'='*60}")
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_clf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    metadata = {
        "version": "2.0_expanded",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "data_source": "cirrhosis.csv only",
        "original_samples": int(df_clean.shape[0]),
        "synthetic_expansion": "3x (3 synthetic per real sample)",
        "total_training_samples": int(X_synthetic.shape[0]),
        "features": FEATURES,
        "target": TARGET,
        "best_model": best_name,
        "results": candidate_results,
        "outlier_caps": caps,
    }
    
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel  → {MODEL_PATH}")
    print(f"Scaler → {SCALER_PATH}")
    print(f"Metadata → {META_PATH}")
    
    # ────────────────────────────────────────────────────────────────────────
    # STEP 10: DETAILED CLASSIFICATION REPORT
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT ({best_name})")
    print(f"{'='*60}")
    
    y_pred_best = best_clf.predict(X_test_scaled)
    print("\n" + classification_report(y_test, y_pred_best, target_names=[f'Stage {i}' for i in range(1, 5)]))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)


if __name__ == "__main__":
    np.random.seed(42)
    train()
