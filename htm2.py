import pandas as pd
import numpy as np
import warnings
import time as timer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import UnivariateSpline
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)


t0 = timer.time()

train = pd.read_csv('/kaggle/input/hack-the-model-stress-prediction-challenge/train_clean.csv')
test = pd.read_csv('/kaggle/input/hack-the-model-stress-prediction-challenge/test_clean.csv')

ID_COL = "id"
TARGET = "Stress (MPa)"

train = train.sort_values("id").reset_index(drop=True)
test_ids = test[ID_COL].values

force_train = train["Force (N)"].values
strain_train = train["Strain (%)"].values
stress_train = train[TARGET].values
force_test = test["Force (N)"].values
strain_test = test["Strain (%)"].values

print(f"Train: {len(train)} rows | Test: {len(test)} rows")
print(f"Data loaded in {timer.time() - t0:.1f}s")

print("\n" + "=" * 60)
print("PHYSICS OPTIMIZATION (Sigmoid Necking)")
print("=" * 60)

t1 = timer.time()


def physics_sigmoid(force, strain, params):
    """
    Engineering stress corrected for necking via sigmoid transition.
    Params: [area0, uts_strain, neck_rate, neck_sharpness]
    """
    area0, uts_strain, neck_rate, neck_sharpness = params
    neck_progress = 1.0 / (1.0 + np.exp(-neck_sharpness * (strain - uts_strain)))
    neck_factor = 1.0 - neck_rate * neck_progress * (np.maximum(strain - uts_strain, 0) / 100.0)
    neck_factor = np.maximum(neck_factor, 0.01)
    return force / (area0 * neck_factor)


def physics_obj(params):
    pred = physics_sigmoid(force_train, strain_train, params)
    return np.sqrt(np.mean((pred - stress_train) ** 2))


bounds_phys = [
    (35.0, 38.5),  
    (10.0, 22.0),   
    (0.01, 1.5),  
    (0.1, 5.0),     
]

res = differential_evolution(
    physics_obj, bounds_phys,
    seed=42, maxiter=500, tol=1e-12,
    mutation=(0.5, 1.5), recombination=0.9,
    popsize=30, workers=-1,  
)
best_params = res.x
best_rmse = res.fun

res2 = minimize(physics_obj, best_params, method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12})
if res2.fun < best_rmse:
    best_params = res2.x
    best_rmse = res2.fun

AREA0 = best_params[0]

physics_train = physics_sigmoid(force_train, strain_train, best_params)
physics_test = physics_sigmoid(force_test, strain_test, best_params)

residual_train = stress_train - physics_train

print(f"Physics RMSE: {best_rmse:.5f}")
print(f"Optimized area0={AREA0:.4f}, uts_strain={best_params[1]:.4f}")
print(f"Physics optimization took {timer.time() - t1:.1f}s")

t2 = timer.time()

spline_stress = UnivariateSpline(
    train["id"].values, stress_train, s=len(train) * 0.5, k=4
)
spline_resid = UnivariateSpline(
    train["id"].values, residual_train, s=len(train) * 2, k=3
)

spline_stress_train = spline_stress(train["id"].values)
spline_stress_test = spline_stress(test_ids)
spline_resid_train = spline_resid(train["id"].values)
spline_resid_test = spline_resid(test_ids)

print(f"Splines fitted in {timer.time() - t2:.1f}s")


t3 = timer.time()


def build_features(force, strain, pos, time_col, ids,
                   phys_pred, spl_stress, spl_resid):
    
    X = {}

    eng_stress = force / AREA0
    X["physics"] = phys_pred
    X["eng_stress"] = eng_stress
    X["physics_minus_eng"] = phys_pred - eng_stress  
    X["spline_stress"] = spl_stress
    X["spline_residual"] = spl_resid
    X["spline_vs_physics"] = spl_stress - phys_pred

    X["force"] = force
    X["strain"] = strain
    X["position"] = pos
    X["time"] = time_col

    id_norm = ids / 5100.0
    X["id_norm"] = id_norm
    X["id_sq"] = id_norm ** 2

    eng_strain_frac = strain / 100.0
    X["true_stress"] = eng_stress * (1 + eng_strain_frac)
    X["true_strain"] = np.log1p(eng_strain_frac)
    X["stiffness"] = force / (strain + 1e-6)

    X["force_x_strain"] = force * strain
    X["force_div_strain"] = force / (strain + 1e-6)

    X["is_elastic"] = (strain < 2.0).astype(np.float32)
    X["is_hardening"] = ((strain >= 5.0) & (strain < 18.0)).astype(np.float32)
    X["is_necking"] = (strain >= 18.0).astype(np.float32)

    X["above_uts"] = np.maximum(0, strain - best_params[1])
    X["above_uts_sq"] = X["above_uts"] ** 2
    X["near_fracture"] = np.maximum(0, strain - 23.0)

    eps = 0.01
    stress_plus = physics_sigmoid(force, strain + eps, best_params)
    stress_minus = physics_sigmoid(force, np.maximum(strain - eps, 0), best_params)
    X["d_stress_d_strain"] = (stress_plus - stress_minus) / (2 * eps)

    X["force_log"] = np.log1p(force)
    X["strain_log"] = np.log1p(strain)

    return pd.DataFrame(X)


X_train = build_features(
    force_train, strain_train,
    train["Position (mm)"].values, train["Time (min)"].values,
    train["id"].values,
    physics_train, spline_stress_train, spline_resid_train
)
X_test = build_features(
    force_test, strain_test,
    test["Position (mm)"].values, test["Time (min)"].values,
    test_ids,
    physics_test, spline_stress_test, spline_resid_test
)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

y = residual_train

print(f"Features: {X_train.shape[1]} (built in {timer.time() - t3:.1f}s)")


print("\n" + "=" * 60)
print("TRAINING 3 DIVERSE LGBMs × 5 FOLDS = 15 fits")
print("=" * 60)

t4 = timer.time()


model_configs = [
    {
        "name": "LGB-deep",
        "params": dict(
            n_estimators=5000, learning_rate=0.005,
            num_leaves=127, subsample=0.85, colsample_bytree=0.8,
            reg_alpha=0.05, reg_lambda=0.1, min_child_samples=3,
            max_bin=511, random_state=42, verbose=-1, n_jobs=-1,
        ),
    },
    {
        "name": "LGB-medium",
        "params": dict(
            n_estimators=3000, learning_rate=0.01,
            num_leaves=63, subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.1, reg_lambda=0.5, min_child_samples=10,
            max_bin=255, random_state=123, verbose=-1, n_jobs=-1,
        ),
    },
    {
        "name": "LGB-regularized",
        "params": dict(
            n_estimators=4000, learning_rate=0.008,
            num_leaves=200, subsample=0.9, colsample_bytree=0.85,
            reg_alpha=0.02, reg_lambda=0.05, min_child_samples=5,
            max_bin=511, random_state=789, verbose=-1, n_jobs=-1,
        ),
    },
]

N_FOLDS = 5
n_models = len(model_configs)

oof_preds = np.zeros((len(X_train), n_models))
test_preds_cv = np.zeros((len(X_test), n_models))  
test_preds_full = np.zeros((len(X_test), n_models)) 

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
folds = list(kf.split(X_train))  

for i, cfg in enumerate(model_configs):
    name = cfg["name"]
    fold_scores = []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        model = LGBMRegressor(**cfg["params"])

        model.fit(
            X_train.iloc[tr_idx], y[tr_idx],
            eval_set=[(X_train.iloc[val_idx], y[val_idx])],
            callbacks=[
               
                __import__('lightgbm').early_stopping(100, verbose=False),
                __import__('lightgbm').log_evaluation(0),
            ],
        )

        oof_preds[val_idx, i] = model.predict(X_train.iloc[val_idx])
        test_preds_cv[:, i] += model.predict(X_test) / N_FOLDS

        rmse = np.sqrt(mean_squared_error(y[val_idx], oof_preds[val_idx, i]))
        fold_scores.append(rmse)

    model_full = LGBMRegressor(**cfg["params"])
    model_full.fit(X_train, y)
    test_preds_full[:, i] = model_full.predict(X_test)

    cv_rmse = np.mean(fold_scores)
    print(f"  [{i+1}/{n_models}] {name}: CV RMSE = {cv_rmse:.5f} ± {np.std(fold_scores):.5f}")

print(f"\nML training took {timer.time() - t4:.1f}s")

print("\n" + "=" * 60)
print("ENSEMBLE & FINAL PREDICTIONS")
print("=" * 60)

from itertools import product

best_ensemble_rmse = float('inf')
best_weights = None

for w0 in np.arange(0.2, 0.6, 0.05):
    for w1 in np.arange(0.1, 0.5, 0.05):
        w2 = 1.0 - w0 - w1
        if w2 < 0.05 or w2 > 0.6:
            continue
        weights = np.array([w0, w1, w2])
        oof_blend = oof_preds @ weights
        rmse = np.sqrt(mean_squared_error(y, oof_blend))
        if rmse < best_ensemble_rmse:
            best_ensemble_rmse = rmse
            best_weights = weights

print(f"Optimal weights: {np.round(best_weights, 3)}")
print(f"Ensemble residual CV RMSE: {best_ensemble_rmse:.5f}")

test_blend = 0.4 * (test_preds_cv @ best_weights) + 0.6 * (test_preds_full @ best_weights)

oof_final = physics_train + oof_preds @ best_weights
test_final = physics_test + test_blend

final_oof_rmse = np.sqrt(mean_squared_error(stress_train, oof_final))
print(f"\n★ FINAL OOF RMSE: {final_oof_rmse:.5f} ★")

stress_min = stress_train.min()
stress_max = stress_train.max()
test_final = np.clip(test_final, stress_min * 0.9, stress_max * 1.05)

submission = pd.DataFrame({
    "id": test_ids,
    "Stress (MPa)": np.round(test_final, 2)
})
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

total_time = timer.time() - t0

print(f"\n{'=' * 60}")
print(f"✓ SUBMISSION SAVED — {len(submission)} rows")
print(f"  Predictions — Mean: {test_final.mean():.2f}, Std: {test_final.std():.2f}")
print(f"  Range: [{test_final.min():.2f}, {test_final.max():.2f}]")
print(f"  Total runtime: {total_time/60:.1f} minutes")
print(f"{'=' * 60}")
