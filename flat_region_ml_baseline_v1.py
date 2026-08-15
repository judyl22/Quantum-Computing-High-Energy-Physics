import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

six_qubits = [
    'IXXIII', 'IXIXII', 'IXIIXI', 'IIXXII', 'IIXIXI', 'IIIXXI'
]

eight_qubits = [
    'IXXIIIII', 'IXIXIIII', 'IXIIXIII', 'IXIIIXII', 'IXIIIIXI',
    'IIXXIIII', 'IIXIXIII', 'IIXIIXII', 'IIXIIIXI', 'IIIXXIII',
    'IIIXIXII', 'IIIXIIXI', 'IIIIXXII', 'IIIIXIXI', 'IIIIIXXI'
]

all_observables = six_qubits + eight_qubits

positive_obs = {
    'IXXIII', 'IIIXXI',
    'IXXIIIII', 'IIIXXIII', 'IIIIIXXI'
}

# 1 bare eye flat regions, 0 otherwise
labels = {}
for obs in all_observables:
    labels[obs] = 1 if obs in positive_obs else 0

def compute_curve_features(T, y):
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)

    # avoid too long
    if len(T) < 3:
        return np.zeros(8)

    slope = np.diff(y) / np.diff(T)
    slope = np.asarray(slope, dtype=float)

    # second devirative (curvature)
    if len(slope) >= 2:
        curvature = np.diff(slope) / np.diff(T[:-1])
    else:
        curvature = np.array([0.0])

    # slope features
    mean_slope = np.mean(np.abs(slope))
    std_slope = np.std(slope)
    max_abs_slope = np.max(np.abs(slope)) if len(slope) > 0 else 0.0
    range_y = np.max(y) - np.min(y)
    mean_y = np.mean(y)
    std_y = np.std(y)

    # "flat-like" score: how much of the curve has slope below a threshold
    flat_fraction = np.mean(np.abs(slope) < 1e-3) if len(slope) > 0 else 0.0

    # "roughness" score: average absolute change in y
    roughness = np.mean(np.abs(np.diff(y)))

    # "plateau-like" score: how much of the curve has very low slope
    plateau_like = np.mean(np.abs(slope) < 5e-4) if len(slope) > 0 else 0.0

    feats = [
        mean_slope,
        std_slope,
        max_abs_slope,
        range_y,
        mean_y,
        std_y,
        flat_fraction,
        plateau_like
    ]
    return np.array(feats, dtype=float)


from fixed_T_simulation import return_vals

X = []
y = []
names = []

for obs in all_observables:
    T_vals, y_vals = return_vals(
        obs,
        n=len(obs),
        N=20,
        g=1.5,
        num_delta_t=2,
        total_time=100
    )

    feat = compute_curve_features(T_vals, y_vals)
    X.append(feat)
    y.append(labels[obs])
    names.append(obs)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape)
print("Labels:", y)
print("Observables:", names)

clf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

cv_scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
print("CV accuracy:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

clf.fit(X, y)

importances = clf.feature_importances_
feature_names = [
    'mean_abs_slope',
    'std_slope',
    'max_abs_slope',
    'y_range',
    'mean_y',
    'std_y',
    'flat_fraction',
    'plateau_like'
]

print("\nFeature importance:")
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")

for obs, arr in zip(names, X):
    pred = clf.predict(arr.reshape(1, -1))[0]
    print(f"{obs}: label={labels[obs]}, pred={pred}")