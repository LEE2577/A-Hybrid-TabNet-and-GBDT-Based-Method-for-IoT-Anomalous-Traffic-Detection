import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import gc
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# 数据处理
print("清洗数据中")
df = pd.read_csv('IoT_Intrusion.csv')
def clean_dataset(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df
df = clean_dataset(df)

target_col = 'label'
label_group_map = {
    'DDoS-ICMP_Flood': 'DDoS', 'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS', 'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',

    'DoS-UDP_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',

    'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',

    'Recon-HostDiscovery': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon', 'VulnerabilityScan': 'Recon',

    'SqlInjection': 'Web', 'XSS': 'Web', 'Backdoor_Malware': 'Web',
    'BrowserHijacking': 'Web', 'CommandInjection': 'Web', 'Uploading_Attack': 'Web',

    'DictionaryBruteForce': 'BruteForce',
    'MITM-ArpSpoofing': 'Spoofing', 'DNS_Spoofing': 'Spoofing',
    'BenignTraffic': 'Benign'
}
df[target_col] = df[target_col].map(label_group_map).fillna(df[target_col])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df[target_col])
feature_names = df.drop(columns=[target_col]).columns.tolist()
X_values = df.drop(columns=[target_col]).values

print(f"标签分布: {Counter(y_encoded)}")
del df  
gc.collect()

print("切分数据 (Train/Val/Test)")
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    X_values, y_encoded, test_size=0.3, random_state=seed, stratify=y_encoded
)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
)
print(f"训练集: {X_train_raw.shape}, 验证集: {X_val_raw.shape}, 测试集: {X_test_raw.shape}")

print("\nGBDT模型：")
#LightGBM
print("训练 LightGBM： ")
lgbm = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    objective='multiclass',
    class_weight='balanced',
    random_state=seed,
    n_jobs=-1
)
lgbm.fit(
    X_train_raw, y_train_raw,
    eval_set=[(X_val_raw, y_val)],
    feature_name=feature_names,
    eval_metric='multi_logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(200)]
)

# XGBoost
print("训练 XGBoost：")
# sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_raw)

xgb_clf = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective='multi:softprob',
    tree_method='hist',
    device='cuda',
    random_state=seed,
    n_jobs=-1,
    early_stopping_rounds=50
)
xgb_clf.fit(
    X_train_raw, y_train_raw,
    sample_weight=sample_weights,
    eval_set=[(X_val_raw, y_val)],
    verbose=200
)

#TabNet
print("\nTabNet：")
#SMOTE + UnderSampling
original_counts = Counter(y_train_raw)
min_limit = 2000  
max_limit = 50000  
sampling_strategy_over = {cls: min_limit for cls, count in original_counts.items() if count < min_limit}
sampling_strategy_under = {cls: max_limit for cls, count in original_counts.items() if count > max_limit}

pipeline_steps = []
if sampling_strategy_over:
    pipeline_steps.append(('o', SMOTE(sampling_strategy=sampling_strategy_over, random_state=seed)))
if sampling_strategy_under:
    pipeline_steps.append(('u', RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=seed)))

if pipeline_steps:
    print("SMOTE + UnderSampling：")
    resample_pipe = Pipeline(steps=pipeline_steps)
    X_train_tab, y_train_tab = resample_pipe.fit_resample(X_train_raw, y_train_raw)
else:
    print("无需重采样")
    X_train_tab, y_train_tab = X_train_raw, y_train_raw

print(f"TabNet 训练集形状: {X_train_tab.shape}")

scaler = StandardScaler()
X_train_tab = scaler.fit_transform(X_train_tab)
X_val_tab = scaler.transform(X_val_raw)
X_test_tab = scaler.transform(X_test_raw)

tabnet_params = dict(
    n_d=64, n_a=64, n_steps=5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',
    device_name='auto'
)

print("TabNet 预训练：")
unsupervised_model = TabNetPretrainer(**tabnet_params)
unsupervised_model.fit(
    X_train=X_train_tab,
    eval_set=[X_val_tab],
    max_epochs=15,
    patience=5,
    batch_size=2048,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
    pretraining_ratio=0.5
)
print("训练 TabNet：")
clf_tabnet = TabNetClassifier(**tabnet_params)

clf_tabnet.fit(
    X_train=X_train_tab, y_train=y_train_tab,
    eval_set=[(X_val_tab, y_val)],
    eval_name=['valid'],
    eval_metric=['accuracy'],
    max_epochs=50,
    patience=15,
    batch_size=2048,
    virtual_batch_size=256,
    num_workers=0,
    drop_last=False,
    from_unsupervised=unsupervised_model # 加载预训练权重
)

from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss
import shap

# 动态权重融合
print("\n寻找最佳融合权重：")
probs_val_lgbm = lgbm.predict_proba(X_val_raw)
probs_val_xgb = xgb_clf.predict_proba(X_val_raw)
probs_val_tabnet = clf_tabnet.predict_proba(X_val_tab)
def loss_func(weights):
    weights = np.array(weights)
    weights /= weights.sum()
    final_probs = (weights[0] * probs_val_lgbm +
                   weights[1] * probs_val_xgb +
                   weights[2] * probs_val_tabnet)
    final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
    return log_loss(y_val, final_probs)
init_weights = [1 / 3, 1 / 3, 1 / 3]
bounds = [(0, 1)] * 3
constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})

res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

best_weights = res.x / res.x.sum()
print(f"最佳权重 -> LGBM: {best_weights[0]:.4f}, XGB: {best_weights[1]:.4f}, TabNet: {best_weights[2]:.4f}")

probs_test_lgbm = lgbm.predict_proba(X_test_raw)
probs_test_xgb = xgb_clf.predict_proba(X_test_raw)
probs_test_tab = clf_tabnet.predict_proba(X_test_tab)
final_test_probs = (best_weights[0] * probs_test_lgbm +
                    best_weights[1] * probs_test_xgb +
                    best_weights[2] * probs_test_tab)

final_preds = np.argmax(final_test_probs, axis=1)

acc = accuracy_score(y_test, final_preds)
macro_f1 = f1_score(y_test, final_preds, average='macro')
weighted_f1 = f1_score(y_test, final_preds, average='weighted')

print("融合模型评估:")
print(f"Accuracy    : {acc:.4f}")
print(f"Macro F1    : {macro_f1:.4f} ")
print(f"Weighted F1 : {weighted_f1:.4f} ")
print("-" * 60)
print("详细分类报告:\n")
print(classification_report(y_test, final_preds, target_names=label_encoder.classes_))

import shap
import matplotlib.pyplot as plt
import os

# SHAP
print("\nSHAP 分析：")
sample_idx = np.random.choice(X_test_raw.shape[0], 1000, replace=False)
X_shap_raw = X_test_raw[sample_idx]
X_shap_tab = X_test_tab[sample_idx] 
if not os.path.exists('results'):
    os.makedirs('results')
#LightGBM
print("LightGBM:")
explainer_lgbm = shap.TreeExplainer(lgbm)
shap_values_lgbm = explainer_lgbm.shap_values(X_shap_raw)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_lgbm, X_shap_raw, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('results/LGBM.png', dpi=300, bbox_inches='tight')
plt.show()

# TabNet
print("TabNet:")
feat_importances = clf_tabnet.feature_importances_
indices = np.argsort(feat_importances)[::-1]
top_k = 20
top_indices = indices[:top_k]
top_importances = feat_importances[top_indices]
top_names = [feature_names[i] for i in top_indices]

plt.figure(figsize=(12, 8))
plt.bar(range(top_k), top_importances, align='center', color='skyblue')
plt.xticks(range(top_k), top_names, rotation=45, ha='right')
plt.ylabel("Importance Score")
plt.xlabel("feature")
plt.tight_layout()
plt.savefig('results/TabNet.png', dpi=300, bbox_inches='tight')

plt.show()

