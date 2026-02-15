import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from scipy.optimize import differential_evolution, minimize
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

train = pd.read_csv('/kaggle/input/hack-the-model-stress-prediction-challenge/train_clean.csv')
test = pd.read_csv('/kaggle/input/hack-the-model-stress-prediction-challenge/test_clean.csv')

ID_COL = "id"
TARGET = "Stress (MPa)"

train_sorted = train.sort_values("id").reset_index(drop=True)


def physics_predict_multi(force, strain, params):
    """Multi-regime physics model with smooth transitions."""
    area0, E_slope, yield_strain, hardening_rate, hardening_exp, \
        uts_strain, neck_rate, neck_exp = params
    
    neck_factor = np.where(
        strain < uts_strain,
        1.0,
        1.0 - neck_rate * ((strain - uts_strain) / 100) ** neck_exp
    )
    neck_factor = np.maximum(neck_factor, 0.01)
    eff_area = area0 * neck_factor
    
    stress = force / eff_area
    
    return stress

def objective_multi(params):
    pred = physics_predict_multi(
        train["Force (N)"].values,
        train["Strain (%)"].values,
        params
    )
    return np.sqrt(np.mean((pred - train[TARGET].values) ** 2))

print("=" * 60)
print("MULTI-REGIME PHYSICS OPTIMIZATION")
print("=" * 60)

best_rmse = float('inf')
best_params = None

bounds = [
    (35.0, 38.0),   
    (0.1, 50.0),   
    (0.5, 5.0),    
    (0.01, 2.0),    
    (0.5, 3.0),    
    (10.0, 35.0),   
    (0.01, 1.0),    
    (0.5, 3.0),     
]

for seed in [42, 123, 456, 789, 2024]:
    result = differential_evolution(
        objective_multi, bounds, seed=seed, 
        maxiter=300, workers=1, tol=1e-10,
        mutation=(0.5, 1.5), recombination=0.9, popsize=25
    )
    if result.fun < best_rmse:
        best_rmse = result.fun
        best_params = result.x
        print(f"  Seed {seed}: RMSE={result.fun:.5f}")

result_nm = minimize(
    objective_multi, best_params, method='Nelder-Mead',
    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
)
if result_nm.fun < best_rmse:
    best_rmse = result_nm.fun
    best_params = result_nm.x

result_pw = minimize(
    objective_multi, best_params, method='Powell',
    options={'maxiter': 5000, 'xtol': 1e-10, 'ftol': 1e-10}
)
if result_pw.fun < best_rmse:
    best_rmse = result_pw.fun
    best_params = result_pw.x

AREA0, E_SLOPE, YIELD_STRAIN, HARD_RATE, HARD_EXP, \
    UTS_STRAIN, NECK_RATE, NECK_EXP = best_params

print(f"\nPhysics RMSE: {best_rmse:.5f}")
print(f"  AREA0={AREA0:.4f}, UTS_STRAIN={UTS_STRAIN:.4f}")
print(f"  NECK_RATE={NECK_RATE:.4f}, NECK_EXP={NECK_EXP:.4f}")

def physics_simple(force, strain, area, neck, thresh):
    eff_area = np.where(
        strain < thresh, area,
        area * (1 - neck * ((strain - thresh) / 100))
    )
    eff_area = np.maximum(eff_area, 1e-6)
    return force / eff_area

def obj_simple(params):
    area, neck, thresh = params
    pred = physics_simple(train["Force (N)"].values, train["Strain (%)"].values, 
                          area, neck, thresh)
    return np.sqrt(np.mean((pred - train[TARGET].values) ** 2))

res_s = differential_evolution(obj_simple, [(35, 38), (0, 0.5), (10, 35)], 
                                seed=42, maxiter=200, workers=1)
AREA_S, NECK_S, THRESH_S = res_s.x
res_s2 = minimize(obj_simple, [AREA_S, NECK_S, THRESH_S], method='Nelder-Mead',
                  options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
if res_s2.fun < res_s.fun:
    AREA_S, NECK_S, THRESH_S = res_s2.x
print(f"Simple Physics RMSE: {min(res_s.fun, res_s2.fun):.5f}")

train["physics_multi"] = physics_predict_multi(
    train["Force (N)"], train["Strain (%)"], best_params)
test["physics_multi"] = physics_predict_multi(
    test["Force (N)"], test["Strain (%)"], best_params)

train["physics_simple"] = physics_simple(
    train["Force (N)"], train["Strain (%)"], AREA_S, NECK_S, THRESH_S)
test["physics_simple"] = physics_simple(
    test["Force (N)"], test["Strain (%)"], AREA_S, NECK_S, THRESH_S)

if np.sqrt(mean_squared_error(train[TARGET], train["physics_multi"])) < \
   np.sqrt(mean_squared_error(train[TARGET], train["physics_simple"])):
    train["physics"] = train["physics_multi"]
    test["physics"] = test["physics_multi"]
    print("Using multi-regime physics as primary")
else:
    train["physics"] = train["physics_simple"]
    test["physics"] = test["physics_simple"]
    print("Using simple physics as primary")

train["residual"] = train[TARGET] - train["physics"]

def make_features(df, is_train=True):
    X = pd.DataFrame(index=df.index)
    
    force = df["Force (N)"].values
    strain = df["Strain (%)"].values
    pos = df["Position (mm)"].values
    time = df["Time (min)"].values
    
    stress_multi = physics_predict_multi(force, strain, best_params)
    stress_simple = physics_simple(force, strain, AREA_S, NECK_S, THRESH_S)
    
    X["physics_multi"] = stress_multi
    X["physics_simple"] = stress_simple
    X["physics_diff"] = stress_multi - stress_simple  
    
    X["eng_stress"] = force / AREA0

    X["force"] = force
    X["strain"] = strain
    X["pos"] = pos
    X["time"] = time
    
    X["force_sq"] = force ** 2
    X["force_cu"] = force ** 3
    X["strain_sq"] = strain ** 2
    X["strain_cu"] = strain ** 3
    X["strain_4"] = strain ** 4
    X["strain_5"] = strain ** 5
    
    X["log1p_force"] = np.log1p(np.abs(force))
    X["log1p_strain"] = np.log1p(strain)
    X["sqrt_strain"] = np.sqrt(np.maximum(strain, 0))
    X["sqrt_force"] = np.sqrt(np.abs(force))
    
    X["force_x_strain"] = force * strain
    X["force_x_strain_sq"] = force * strain**2
    X["force_sq_x_strain"] = force**2 * strain
    X["force_div_strain"] = force / (strain + 1e-6)
    X["strain_div_force"] = strain / (np.abs(force) + 1e-6)
    X["pos_x_force"] = pos * force
    X["pos_x_strain"] = pos * strain
    X["time_x_force"] = time * force
    X["time_x_strain"] = time * strain
    
    for thresh in [THRESH_S, UTS_STRAIN, 5.0, 15.0, 20.0, 24.0]:
        name = f"t{thresh:.0f}"
        above = np.maximum(0, strain - thresh)
        below = np.minimum(strain, thresh)
        X[f"is_above_{name}"] = (strain > thresh).astype(float)
        X[f"above_{name}"] = above
        X[f"above_sq_{name}"] = above ** 2
        X[f"above_cu_{name}"] = above ** 3
        X[f"below_{name}"] = below
        X[f"below_sq_{name}"] = below ** 2
        X[f"force_x_above_{name}"] = force * above
        X[f"force_x_is_{name}"] = force * (strain > thresh).astype(float)
        X[f"ratio_{name}"] = strain / (thresh + 1e-6)
    
    X["eff_area_multi"] = force / (stress_multi + 1e-6)
    X["eff_area_simple"] = force / (stress_simple + 1e-6)
    X["area_ratio_multi"] = X["eff_area_multi"] / AREA0
    X["area_ratio_simple"] = X["eff_area_simple"] / AREA0
    X["area_reduction"] = 1 - X["area_ratio_multi"]
    
    X["stress_x_strain"] = stress_multi * strain
    X["stress_x_strain_sq"] = stress_multi * strain**2
    X["stress_sq_x_strain"] = stress_multi**2 * strain
    X["stress_div_strain"] = stress_multi / (strain + 1e-6)
    X["stress_sq"] = stress_multi ** 2
    X["stress_cu"] = stress_multi ** 3
    
    id_vals = df["id"].values
    X["id_norm"] = id_vals / 5000.0  
    X["id_sq"] = (id_vals / 5000.0) ** 2
    
    X["near_fracture"] = np.maximum(0, (id_vals - 4800) / 200)
    X["fracture_indicator"] = (id_vals > 4900).astype(float)
    X["force_x_fracture"] = force * X["fracture_indicator"]
    
    eps = 0.01
    stress_plus = physics_predict_multi(force, strain + eps, best_params)
    stress_minus = physics_predict_multi(force, np.maximum(strain - eps, 0), best_params)
    X["d_stress_d_strain"] = (stress_plus - stress_minus) / (2 * eps)
    
    X["pos_minus_time"] = pos - time
    X["pos_div_time"] = pos / (time + 1e-6)
    
    X["stress_multi_log"] = np.log1p(np.maximum(stress_multi, 0))
    
    X["force_to_max_force_ratio"] = force / 15400.0  
    X["strain_to_max_strain_ratio"] = strain / 25.5   
    
    return X

print(f"\nBuilding features...")
X_train = make_features(train, is_train=True)
X_test = make_features(test, is_train=False)
y = train["residual"]

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

const_cols = [c for c in X_train.columns if X_train[c].std() < 1e-10]
if const_cols:
    X_train = X_train.drop(columns=const_cols)
    X_test = X_test.drop(columns=const_cols)

print(f"Features: {X_train.shape[1]}")


def get_lgb_1():
    """Deep, slow learner"""
    return LGBMRegressor(
        n_estimators=8000, learning_rate=0.003,
        num_leaves=127, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.3, min_child_samples=5,
        random_state=42, verbose=-1, n_jobs=-1
    )

def get_lgb_2():
    """Medium depth, moderate regularization"""
    return LGBMRegressor(
        n_estimators=6000, learning_rate=0.005,
        num_leaves=80, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.2, reg_lambda=0.5, min_child_samples=10,
        random_state=123, verbose=-1, n_jobs=-1
    )

def get_lgb_3():
    """Shallow, heavily regularized for diversity"""
    return LGBMRegressor(
        n_estimators=5000, learning_rate=0.008,
        num_leaves=50, subsample=0.75, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=1.5, min_child_samples=20,
        random_state=789, verbose=-1, n_jobs=-1
    )

def get_lgb_4():
    """Different seed, wide trees"""
    return LGBMRegressor(
        n_estimators=7000, learning_rate=0.004,
        num_leaves=100, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=0.05, reg_lambda=0.2, min_child_samples=8,
        random_state=2024, verbose=-1, n_jobs=-1
    )

def get_xgb_1():
    """Deep XGB"""
    return XGBRegressor(
        n_estimators=8000, learning_rate=0.003,
        max_depth=10, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=0.3, min_child_weight=1,
        tree_method='hist', random_state=42, verbosity=0, n_jobs=-1
    )

def get_xgb_2():
    """Moderate XGB"""
    return XGBRegressor(
        n_estimators=5000, learning_rate=0.007,
        max_depth=7, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.3, reg_lambda=1.0, min_child_weight=5,
        tree_method='hist', random_state=456, verbosity=0, n_jobs=-1
    )

def get_xgb_3():
    """Shallow regularized XGB"""
    return XGBRegressor(
        n_estimators=4000, learning_rate=0.01,
        max_depth=5, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=3.0, min_child_weight=15,
        tree_method='hist', random_state=999, verbosity=0, n_jobs=-1
    )

def get_cat_1():
    """CatBoost for diversity — different algorithm entirely"""
    return CatBoostRegressor(
        iterations=6000, learning_rate=0.005,
        depth=8, l2_leaf_reg=3, subsample=0.85,
        random_seed=42, verbose=0
    )

def get_cat_2():
    """Shallow CatBoost"""
    return CatBoostRegressor(
        iterations=4000, learning_rate=0.01,
        depth=6, l2_leaf_reg=5, subsample=0.75,
        random_seed=123, verbose=0
    )

models = [
    get_lgb_1(), get_lgb_2(), get_lgb_3(), get_lgb_4(),
    get_xgb_1(), get_xgb_2(), get_xgb_3(),
    get_cat_1(), get_cat_2()
]
n_models = len(models)

N_FOLDS = 10
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train), n_models))
test_preds = np.zeros((len(X_test), n_models))

print(f"\n{'='*60}")
print(f"TRAINING {n_models} MODELS x {N_FOLDS} FOLDS")
print(f"{'='*60}")

model_names = [
    "LGB-deep", "LGB-med", "LGB-shallow", "LGB-wide",
    "XGB-deep", "XGB-med", "XGB-shallow",
    "CAT-deep", "CAT-shallow"
]

for i, (model_fn, name) in enumerate(zip(
    [get_lgb_1, get_lgb_2, get_lgb_3, get_lgb_4,
     get_xgb_1, get_xgb_2, get_xgb_3,
     get_cat_1, get_cat_2],
    model_names
)):
    print(f"\n[{i+1}/{n_models}] {name}")
    fold_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        model = model_fn()  
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx, i] = model.predict(X_val)
        test_preds[:, i] += model.predict(X_test) / N_FOLDS
        
        rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx, i]))
        fold_scores.append(rmse)
    
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"  → CV RMSE: {mean_score:.5f} ± {std_score:.5f}")

print(f"\n{'='*60}")
print("LEVEL-2 META-LEARNING")
print(f"{'='*60}")

meta = RidgeCV(
    alphas=np.logspace(-3, 3, 50), cv=5
)
meta.fit(oof_preds, y)
oof_meta = meta.predict(oof_preds)
residual_rmse = np.sqrt(mean_squared_error(y, oof_meta))
print(f"Ridge (alpha={meta.alpha_:.4f}): Residual RMSE = {residual_rmse:.5f}")
print(f"Ridge weights: {np.round(meta.coef_, 4)}")

train_final = train["physics"].values + oof_meta
final_rmse = np.sqrt(mean_squared_error(train[TARGET], train_final))

print(f"\n{'='*60}")
print(f"★ FINAL OOF RMSE: {final_rmse:.5f} ★")
print(f"{'='*60}")

print("\nRetraining all models on full data...")
full_test_preds = np.zeros((len(X_test), n_models))

model_fns = [get_lgb_1, get_lgb_2, get_lgb_3, get_lgb_4,
             get_xgb_1, get_xgb_2, get_xgb_3,
             get_cat_1, get_cat_2]

for i, fn in enumerate(model_fns):
    model = fn()
    model.fit(X_train, y)
    full_test_preds[:, i] = model.predict(X_test)

blended_test = 0.5 * test_preds + 0.5 * full_test_preds
test_meta = meta.predict(blended_test)
final_test = test["physics"].values + test_meta

submission = pd.DataFrame({
    "id": test[ID_COL],
    "Stress (MPa)": np.round(final_test, 2)
})
submission.to_csv("submission.csv", index=False)

print(f"\n✓ SUBMISSION SAVED")
print(f"  Predictions — Mean: {final_test.mean():.2f}, Std: {final_test.std():.2f}")
print(f"  Min: {final_test.min():.2f}, Max: {final_test.max():.2f}")
