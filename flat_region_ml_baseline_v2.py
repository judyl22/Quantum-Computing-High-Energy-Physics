import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from fixed_T_simulation import return_vals

# Quick baseline: label a small set of observables by eye.
# 1 = clear flat/plateau region, 0 = no clear flat region.
# This version intentionally uses a small, fast dataset so it
# can finish quickly in a notebook or local script.
all_observables = [
    'IXXIII',
    'IIIXXI',
    'IIIIIXXI',
    'IIIXXIII',
    'IXXIIIII',
    'IXIXII',
    'IXIIXI',
    'IIXXII',
    'IIXIXI',
    'IIIIXIXI',
]

positive_obs = {
    'IXXIII',
    'IIIXXI',
    'IIIIIXXI',
    'IIIXXIII',
    'IXXIIIII',
}

labels = {obs: 1 if obs in positive_obs else 0 for obs in all_observables}

SIM_CONFIG = {
    'n': 4,
    'N': 10,
    'g': 1.5,
    'num_delta_t': 1,
    'total_time': 20,
}


def compute_curve_features(T, y):
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(T) < 3:
        return np.zeros(8, dtype=float)

    slope = np.diff(y) / np.diff(T)
    if len(slope) >= 2:
        curvature = np.diff(slope) / np.diff(T[:-1])
    else:
        curvature = np.array([0.0])

    mean_abs_slope = np.mean(np.abs(slope)) if len(slope) > 0 else 0.0
    std_slope = np.std(slope) if len(slope) > 0 else 0.0
    max_abs_slope = np.max(np.abs(slope)) if len(slope) > 0 else 0.0
    y_range = np.max(y) - np.min(y)
    mean_y = np.mean(y)
    std_y = np.std(y)

    flat_fraction = np.mean(np.abs(slope) < 1e-3) if len(slope) > 0 else 0.0
    plateau_like = np.mean(np.abs(slope) < 5e-4) if len(slope) > 0 else 0.0
    roughness = np.mean(np.abs(np.diff(y))) if len(y) > 1 else 0.0

    features = np.array([
        mean_abs_slope,
        std_slope,
        max_abs_slope,
        y_range,
        mean_y,
        std_y,
        flat_fraction,
        plateau_like,
    ], dtype=float)

    return features


def build_dataset():
    X = []
    y = []
    names = []

    for obs in all_observables:
        T_vals, y_vals = return_vals(
            obs,
            n=len(obs),
            N=SIM_CONFIG['N'],
            g=SIM_CONFIG['g'],
            num_delta_t=SIM_CONFIG['num_delta_t'],
            total_time=SIM_CONFIG['total_time'],
        )

        feat = compute_curve_features(T_vals, y_vals)
        X.append(feat)
        y.append(labels[obs])
        names.append(obs)

    return np.array(X), np.array(y), names


def main():
    X, y, names = build_dataset()
    print("Dataset shape:", X.shape)
    print("Positive samples:", int(np.sum(y)))
    print("Negative samples:", int(len(y) - np.sum(y)))

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print("Cross-val accuracy:", cv_scores)
    print("Mean CV accuracy:", np.mean(cv_scores))

    clf.fit(X, y)
    feature_names = [
        'mean_abs_slope',
        'std_slope',
        'max_abs_slope',
        'y_range',
        'mean_y',
        'std_y',
        'flat_fraction',
        'plateau_like',
    ]

    print("\nFeature importances:")
    for name, importance in zip(feature_names, clf.feature_importances_):
        print(f"  {name}: {importance:.4f}")

    print("\nPer-observable predictions:")
    for obs, row, true_label in zip(names, X, y):
        pred = clf.predict(row.reshape(1, -1))[0]
        print(f"  {obs}: true={true_label}, pred={pred}")


if __name__ == '__main__':
    main()
