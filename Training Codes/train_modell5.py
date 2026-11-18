# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:27:59 2026

@author: Karthik Pai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Website Trust Score Pipeline (URL-only, zero scraping)
======================================================

Gap 1  Unseen-domain generalization : random vs domain-disjoint vs cross-dataset evaluation
Gap 2  Calibrated trust score       : Platt + isotonic on a held-out GROUPED calibration split,
                                      Brier / ECE / PR-AUC / precision at realistic phishing prevalence
Gap 3  Free-hosting blind spot      : PSL-private tenant parsing, tenant-level features, slice evaluation,
                                      with/without hosting-feature ablation
Gap 4  Latency                      : single-URL model latency + end-to-end (extraction + model) p50/p95
Gap 5  Plain-language explanations  : SHAP aggregated into feature groups -> template sentences

Also included: TP/TN/FP/FN/TPR/TNR for every model, feature selection, PCA/t-SNE, 5-fold CV (random + domain-grouped),
SHAP + LIME, and ~25 comparative / metric graphs (saved in <out>/plots/, displayed when SHOW_PLOTS = True).

Usage
-----
  python trust_pipeline.py train --data PhiUSIIL_Phishing_URL_Dataset.csv --out artifacts
  python trust_pipeline.py train --data PhiUSIIL_Phishing_URL_Dataset.csv --external my_test.csv
  python trust_pipeline.py score --url "https://paypal-login.vercel.app/verify" --artifacts artifacts

Dependencies
------------
  pip install pandas numpy scikit-learn xgboost lightgbm catboost shap lime matplotlib torch tldextract joblib
  pip install transformers   # only if you enable DistilBERT

External test CSV (optional, for the cross-dataset regime): columns `url,label` with label 1 = phishing, 0 = legit.
PhiUSIIL has no timestamps, so a time split needs your own dated feed (PhishTank/OpenPhish + Tranco).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch   # must be imported first on Windows (avoids WinError 127)
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset

import argparse
import json
import math
import os
import re
import textwrap
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tldextract
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             classification_report, cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CONFIG
# ==============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_DISTILBERT = False          # slow on CPU; enable on GPU. Evaluated on the SAME test set as the others.
BERT_MAX_TRAIN = 60000          # cap on training rows for DistilBERT (None = all)
RANDOM_REGIME_MODELS = ["1. Logistic Regression", "2. Random Forest", "3. XGBoost",
                        "4. LightGBM", "5. CatBoost"]   # models also run on the random split (Gap 1 headline)
SHIP_ENSEMBLE = True            # score with calibrated mean(LightGBM, Char-CNN); set False to ship LightGBM alone
DISAGREEMENT_LIMIT = 0.30       # |p_lgbm - p_cnn| above this -> "Limited information"
PHISH_PREVALENCES = [0.05, 0.01]  # deployment priors for precision estimates (95/5 and 99/1)

# --- feature selection / dimensionality reduction ---
USE_SELECTED_FEATURES = False   # True: train ALL models on the selected subset; False: only model "14" uses it
N_SELECTED_FEATURES = 25        # how many features to keep
CORR_THRESHOLD = 0.95           # drop one of two features correlated above this
PCA_VARIANCE = 0.95             # PCA keeps components explaining this much variance
RUN_TSNE = True
TSNE_SAMPLES = 4000
SELECTED_COLS: List[str] = []   # filled at run time

# --- cross-validation ---
CV_FOLDS = 5
CV_MODELS = ["1. Logistic Regression", "3. XGBoost", "4. LightGBM"]   # add more names (e.g. "2. Random Forest") if time allows
CV_MAX_ROWS = 100000            # CV uses at most this many URLs (whole domains sampled); None = all
CV_INCLUDE_DL = False           # True also cross-validates the Char-CNN (slow)

# --- graphs ---
SHOW_PLOTS = True              # True: also display each graph (Spyder Plots pane). False: only save PNGs.
                                # (running from a terminal with True opens windows; use False there)

# --- explainability ---
SHAP_SAMPLES = 2000
RUN_LIME = True
LIME_SAMPLES = 20               # URLs used for the SHAP-vs-LIME agreement score

# --- speed settings for the deep models (raise them if you have a GPU) ---
DL_EPOCHS = 5                   # max epochs (early stopping may stop sooner)
DL_PATIENCE = 2
DL_MAX_TRAIN = 100000           # use at most this many training URLs for deep models (None = all)
SKIP_MODELS = ["8. BiLSTM+Attention"]   # slowest model; remove from this list to include it
torch.set_num_threads(os.cpu_count() or 4)

MAX_CHAR_LEN = 100
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}   # 0 = pad, len(CHARS)+1 = unknown
VOCAB_SIZE = len(CHARS) + 2

# Curated shared-hosting suffixes (union with the PSL private section).
# sites.google.com / ipfs.io / cloudfront.net are omitted: tenant lives in the path, or it is a CDN.
CURATED_SHARED = [
    "vercel.app", "netlify.app", "firebaseapp.com", "web.app", "pages.dev", "workers.dev", "github.io",
    "herokuapp.com", "ngrok.io", "ngrok-free.app", "weebly.com", "weeblysite.com", "blogspot.com",
    "000webhostapp.com", "duckdns.org", "azurewebsites.net", "repl.co", "glitch.me", "onrender.com",
    "r2.dev", "framer.app",
]

TARGET_BRANDS = [
    "paypal", "microsoft", "apple", "google", "amazon", "netflix", "facebook", "instagram", "whatsapp",
    "sbi", "hdfc", "icici", "chase", "wellsfargo", "citibank", "binance", "coinbase", "metamask", "dhl",
    "fedex", "usps", "adobe", "dropbox", "linkedin", "yahoo", "outlook", "office365",
]
INTENT_KEYWORDS = ["login", "signin", "sign-in", "verify", "verification", "secure", "security", "update",
                   "account", "banking", "wallet", "checkout", "payment", "pay", "confirm", "auth", "kyc",
                   "password"]
SUSPICIOUS_TLDS = {"gq", "tk", "ml", "cf", "ga", "xyz", "top", "shop", "club", "online", "site", "live",
                   "icu", "buzz", "ru"}

# Word-boundary matching: 'apple' will not fire on 'pineapple', 'pay' not on 'payroll'.
_BRAND_RES = [re.compile(rf"(?<![a-z]){re.escape(b)}(?![a-z])") for b in TARGET_BRANDS]
_INTENT_RES = [re.compile(rf"(?<![a-z]){re.escape(k)}(?![a-z])") for k in INTENT_KEYWORDS]
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.I)

_PSL_PUB = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
_PSL_BASE = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=False)


# ==============================================================================
# 2. URL PARSING + FEATURES
# ==============================================================================
def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def canonicalize(url: str) -> str:
    u = SCHEME_RE.sub("", str(url).strip())
    return re.sub(r"^www\.", "", u, flags=re.I)


def parse_host(host: str) -> dict:
    """Tenant-aware host parsing. `group` is the registrable domain, or the full tenant host on shared platforms."""
    if IP_RE.match(host):
        return dict(subdomain="", label=host, suffix="", group=host, is_shared=0, is_ip=1)
    for sfx in CURATED_SHARED:
        if host.endswith("." + sfx):
            parts = host[: -(len(sfx) + 1)].split(".")
            return dict(subdomain=".".join(parts[:-1]), label=parts[-1], suffix=sfx,
                        group=f"{parts[-1]}.{sfx}", is_shared=1, is_ip=0)
    e_pub, e_base = _PSL_PUB(host), _PSL_BASE(host)
    if not e_pub.domain:
        return dict(subdomain="", label=host, suffix=e_pub.suffix, group=host, is_shared=0, is_ip=0)
    is_shared = int(e_pub.suffix != e_base.suffix)          # private-PSL suffix = shared platform
    group = f"{e_pub.domain}.{e_pub.suffix}" if e_pub.suffix else e_pub.domain
    return dict(subdomain=e_pub.subdomain, label=e_pub.domain, suffix=e_pub.suffix, group=group,
                is_shared=is_shared, is_ip=0)


def _brands_in(text: str) -> set:
    return {b for b, r in zip(TARGET_BRANDS, _BRAND_RES) if r.search(text)}


def extract_one(url: str) -> dict:
    canon = canonicalize(url)
    try:
        p = urlparse("http://" + canon)
        host, path, query = (p.hostname or "").strip(".").lower(), p.path or "", p.query or ""
    except ValueError:
        host, path, query = canon.split("/")[0].lower(), "", ""
    h = parse_host(host)
    sub, label, sfx = h["subdomain"], h["label"], h["suffix"]
    tld = sfx.split(".")[-1] if sfx else ""
    low = canon.lower()
    L = max(len(canon), 1)
    letters, digits = sum(c.isalpha() for c in canon), sum(c.isdigit() for c in canon)

    brand_sub = int(bool(_brands_in(sub) - {label}))
    brand_label = int(label not in TARGET_BRANDS and bool(_brands_in(label)))   # paypal-login.com, not paypal.com
    brand_path = int(bool(_brands_in(path.lower() + " " + query.lower())))
    intent_n = sum(1 for r in _INTENT_RES if r.search(low))
    risky_ctx = int(intent_n > 0 or brand_sub or brand_label)

    return {
        "group_domain": h["group"], "TLD_cat": tld or "unknown",
        "url_len": len(canon), "host_len": len(host), "path_len": len(path), "query_len": len(query),
        "path_depth": path.count("/"), "has_path": int(len(path) > 1), "has_query": int(len(query) > 0),
        "subdomain_count": len(sub.split(".")) if sub else 0, "sub_len": len(sub), "sub_entropy": shannon_entropy(sub),
        "label_len": len(label), "label_entropy": shannon_entropy(label),
        "label_digit_ratio": sum(c.isdigit() for c in label) / max(len(label), 1), "label_hyphens": label.count("-"),
        "tld_len": len(tld), "is_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_ip": h["is_ip"], "is_shared_host": h["is_shared"],
        "letter_ratio": letters / L, "digit_ratio": digits / L, "special_ratio": (len(canon) - letters - digits) / L,
        "num_dots": canon.count("."), "num_hyphens": canon.count("-"), "num_slashes": canon.count("/"),
        "num_qmark": canon.count("?"), "num_eq": canon.count("="), "num_amp": canon.count("&"),
        "has_at": int("@" in canon), "has_hex": int(bool(re.search(r"%[0-9a-fA-F]{2}", canon))),
        "host_entropy": shannon_entropy(host), "path_entropy": shannon_entropy(path + query),
        "brand_in_sub": brand_sub, "brand_in_label": brand_label, "brand_in_path": brand_path,
        "intent_count": intent_n, "has_intent": int(intent_n > 0),
        "shared_x_risky": h["is_shared"] * risky_ctx,
        "hosting_risk_score": h["is_shared"] * (brand_sub + brand_label + intent_n),
    }


def extract_features(urls) -> pd.DataFrame:
    return pd.DataFrame([extract_one(u) for u in urls])


# Ablation: features that encode the hosting-platform signal
HOSTING_FEATURES = ["is_shared_host", "shared_x_risky"]

# Gap 5: every numeric feature belongs to exactly one human-readable group.
FEATURE_GROUPS = {
    "random_name": ["host_entropy", "sub_entropy", "label_entropy", "label_digit_ratio", "digit_ratio",
                    "label_len", "label_hyphens", "letter_ratio"],
    "free_hosting": ["is_shared_host", "shared_x_risky", "hosting_risk_score"],
    "brand_misuse": ["brand_in_sub", "brand_in_label", "brand_in_path"],
    "sensitive_words": ["intent_count", "has_intent"],
    "odd_ending": ["is_suspicious_tld", "tld_len"],
    "disguise": ["has_at", "has_hex", "is_ip", "special_ratio", "num_eq", "num_amp", "num_qmark", "has_query",
                 "query_len"],
    "link_structure": ["url_len", "host_len", "path_len", "path_depth", "has_path", "subdomain_count", "sub_len",
                       "num_dots", "num_hyphens", "num_slashes", "path_entropy"],
}
GROUP_RISK_TEXT = {
    "random_name": "The web address looks randomly generated, which is common for throwaway scam sites.",
    "free_hosting": "The site sits on a free shared hosting platform, which scammers often use to borrow a trusted name.",
    "brand_misuse": "The address uses a well-known brand name, but this does not look like that brand's official site.",
    "sensitive_words": "The link contains words about logging in, verifying or paying, which phishing pages often use.",
    "odd_ending": "The address ends in a domain type that is frequently used by disposable or scam websites.",
    "disguise": "The link uses tricks that can hide where it really leads (symbols, encoded characters or a raw number address).",
    "link_structure": "The link is unusually long or complicated compared with typical legitimate websites.",
}
GROUP_SAFE_TEXT = {
    "random_name": "The web address looks like a normal, readable name.",
    "free_hosting": "The site is not on a free shared hosting platform.",
    "brand_misuse": "The address does not imitate a well-known brand.",
    "sensitive_words": "The link does not push you toward logging in or paying.",
    "odd_ending": "The address uses an ordinary domain ending.",
    "disguise": "The link has no signs of hidden or disguised destinations.",
    "link_structure": "The link has a simple, ordinary structure.",
}


def numeric_columns(feat_df: pd.DataFrame) -> List[str]:
    return [c for c in feat_df.columns if c not in ("group_domain", "TLD_cat")]


def check_groups(cols: List[str]):
    grouped = [f for fs in FEATURE_GROUPS.values() for f in fs]
    assert sorted(grouped) == sorted(cols), f"Feature/group mismatch: {set(cols) ^ set(grouped)}"


# ==============================================================================
# 3. BATCH CONTAINER, ENCODING, SPLITS
# ==============================================================================
def encode_urls(urls) -> tuple:
    seq = np.zeros((len(urls), MAX_CHAR_LEN), dtype=np.int64)
    lens = np.ones(len(urls), dtype=np.int64)
    unk = len(CHARS) + 1
    for i, u in enumerate(urls):
        s = canonicalize(u).lower()[:MAX_CHAR_LEN]
        if s:
            seq[i, :len(s)] = [CHAR2IDX.get(c, unk) for c in s]
            lens[i] = len(s)
    return torch.from_numpy(seq), torch.from_numpy(lens)


@dataclass
class Batch:
    urls: List[str]
    X: pd.DataFrame              # numeric cols + TLD_cat
    tab: np.ndarray              # scaled numeric features (float32)
    seq: torch.Tensor
    lens: torch.Tensor
    y: Optional[np.ndarray]

    def take(self, idx) -> "Batch":
        idx = np.asarray(idx)
        return Batch([self.urls[i] for i in idx], self.X.iloc[idx], self.tab[idx], self.seq[idx], self.lens[idx],
                     None if self.y is None else self.y[idx])

    def __len__(self):
        return len(self.urls)



def make_batch(urls, feat_df: pd.DataFrame, cols: List[str], scaler: Optional[StandardScaler], y=None,
               encode: bool = True) -> Batch:

    X = feat_df[cols + ["TLD_cat", "group_domain"]].reset_index(drop=True)

    tab = scaler.transform(X[cols]).astype(np.float32) if scaler is not None else np.zeros((len(X), len(cols)), np.float32)

    if encode:
        seq, lens = encode_urls(list(urls))
    else:
        seq, lens = torch.zeros((len(X), 1), dtype=torch.long), torch.ones(len(X), dtype=torch.long)

    return Batch(list(urls), X, tab, seq, lens, None if y is None else np.asarray(y))


@dataclass
class Regime:
    name: str
    cols: List[str]
    scaler: StandardScaler
    fit: Batch
    val: Batch
    cal: Batch
    test: Batch


def grouped_split(groups: np.ndarray, seed: int):
    """Domain-disjoint: test 20% of groups; of the rest, 75% fit / 25% holdout; holdout split evenly val / cal."""
    idx = np.arange(len(groups))
    rest, test = next(GroupShuffleSplit(1, test_size=0.20, random_state=seed).split(idx, groups=groups))
    a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(rest, groups=groups[rest]))
    fit, hold = rest[a], rest[b]
    v, c = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(hold, groups=groups[hold]))
    return fit, hold[v], hold[c], test


def random_split(y: np.ndarray, seed: int):
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    fit, hold = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=seed)
    val, cal = train_test_split(hold, test_size=0.50, stratify=y[hold], random_state=seed)
    return fit, val, cal, test


def build_regime(name: str, urls, feat_df, y, cols, split, encode: bool = True) -> Regime:
    fit, val, cal, test = split
    scaler = StandardScaler().fit(feat_df.iloc[fit][cols])
    full = make_batch(urls, feat_df, cols, scaler, y, encode=encode)
    return Regime(name, cols, scaler, full.take(fit), full.take(val), full.take(cal), full.take(test))


# ==============================================================================
# 4. MODELS  (each trainer returns predict_fn(Batch) -> P(phishing); .model holds the fitted object)
# ==============================================================================
def _wrap(fn, model):
    fn.model = model
    return fn


def train_logreg(reg: Regime):
    m = LogisticRegression(max_iter=2000, random_state=SEED).fit(reg.fit.tab, reg.fit.y)
    return _wrap(lambda b: m.predict_proba(b.tab)[:, 1], m)


def train_rf(reg: Regime):
    m = RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_leaf=3, n_jobs=-1,
                               random_state=SEED).fit(reg.fit.X[reg.cols], reg.fit.y)
    return _wrap(lambda b: m.predict_proba(b.X[reg.cols])[:, 1], m)


def train_xgb(reg: Regime):
    m = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2.0, eval_metric="logloss", early_stopping_rounds=50, n_jobs=-1,
                          random_state=SEED)
    m.fit(reg.fit.X[reg.cols], reg.fit.y, eval_set=[(reg.val.X[reg.cols], reg.val.y)], verbose=False)
    return _wrap(lambda b: m.predict_proba(b.X[reg.cols])[:, 1], m)


def train_lightgbm(reg: Regime, drop=()):
    cols = [c for c in reg.cols if c not in drop]
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                           n_jobs=-1, random_state=SEED, verbose=-1)
    m.fit(reg.fit.X[cols], reg.fit.y, eval_set=[(reg.val.X[cols], reg.val.y)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    fn = _wrap(lambda b: m.predict_proba(b.X[cols])[:, 1], m)
    fn.cols = cols
    return fn


def train_catboost(reg: Regime):
    cc = reg.cols + ["TLD_cat"]
    m = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.06, cat_features=["TLD_cat"],
                           early_stopping_rounds=50, verbose=0, random_seed=SEED)
    m.fit(reg.fit.X[cc], reg.fit.y, eval_set=(reg.val.X[cc], reg.val.y))
    return _wrap(lambda b: m.predict_proba(b.X[cc])[:, 1], m)


# ---- deep models (padding-aware) ----
class TabularMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 1))

    def forward(self, seq, lens, tab):
        return self.net(tab).squeeze(-1)


class CharCNN1D(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, emb=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.c3 = nn.Conv1d(emb, 64, 3, padding=1)
        self.c5 = nn.Conv1d(emb, 64, 5, padding=2)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, seq, lens, tab=None):
        mask = (seq != 0).unsqueeze(1)
        x = self.emb(seq).transpose(1, 2)
        pooled = [torch.relu(c(x)).masked_fill(~mask, -1e9).max(dim=-1)[0] for c in (self.c3, self.c5)]
        return self.fc(torch.cat(pooled, 1)).squeeze(-1)


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, emb=64, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hid * 2, 1)
        self.fc = nn.Sequential(nn.Linear(hid * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, seq, lens, tab=None):
        packed = pack_padded_sequence(self.emb(seq), lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=seq.size(1))
        w = torch.softmax(self.attn(h).squeeze(-1).masked_fill(seq == 0, -1e9), dim=1).unsqueeze(-1)
        return self.fc((w * h).sum(1)).squeeze(-1)


class HybridCNNBiLSTM(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, emb=64, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv = nn.Conv1d(emb, 64, 3, padding=1)
        self.lstm = nn.LSTM(64, hid, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(hid * 2, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, seq, lens, tab=None):
        x = torch.relu(self.conv(self.emb(seq).transpose(1, 2))).transpose(1, 2)
        packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=seq.size(1))
        h = h.masked_fill((seq == 0).unsqueeze(-1), -1e9).max(dim=1)[0]      # masked max-pool over time
        return self.fc(h).squeeze(-1)


class TransformerURLEncoder(nn.Module):
    """Lightweight Transformer encoder trained from scratch on char-level URL sequences (Gap model 15)."""
    def __init__(self, vocab=VOCAB_SIZE, d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
                 max_len=MAX_CHAR_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)   # +1 for CLS position 0
        self.cls_emb = nn.Embedding(1, d_model)             # single learned CLS token
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, seq, lens, tab=None):
        B, T = seq.shape
        # prepend learned CLS token (index 0)
        cls = self.cls_emb(torch.zeros(B, 1, dtype=torch.long, device=seq.device))  # (B, 1, d)
        tok = self.tok_emb(seq)                              # (B, T, d)
        pos_ids = torch.arange(1, T + 1, device=seq.device).unsqueeze(0)  # (1, T)
        tok = tok + self.pos_emb(pos_ids)
        x = torch.cat([cls, tok], dim=1)                    # (B, T+1, d)
        # padding mask: True = ignore; CLS never masked
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=seq.device),
            seq == 0
        ], dim=1)                                            # (B, T+1)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(out[:, 0, :]).squeeze(-1)           # pool CLS position


class TabNetGatedMLP(nn.Module):
    """TabNet-style gated MLP with GLU gating per step (Gap model 16)."""
    def __init__(self, in_dim, hidden_dim=128, n_steps=3):
        super().__init__()
        self.n_steps = n_steps
        self.bns = nn.ModuleList([nn.BatchNorm1d(in_dim) for _ in range(n_steps)])
        self.fc_in = nn.ModuleList([nn.Linear(in_dim, 2 * hidden_dim) for _ in range(n_steps)])
        self.shared = nn.Linear(hidden_dim, hidden_dim)
        self.step_specific = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_steps)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, seq, lens, tab):
        x = tab
        agg = None
        for i in range(self.n_steps):
            xn = self.bns[i](x)
            h_gate = self.fc_in[i](xn)                      # (B, 2*hidden)
            h, gate = h_gate.chunk(2, dim=-1)
            h = h * torch.sigmoid(gate)                     # GLU gating
            h = torch.relu(self.shared(h) + self.step_specific[i](h))
            agg = h if agg is None else agg + h
        return self.head(agg).squeeze(-1)


@torch.no_grad()
def predict_torch(model: nn.Module, b: Batch, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(b), bs):
        s, l = b.seq[i:i + bs].to(DEVICE), b.lens[i:i + bs].to(DEVICE)
        t = torch.from_numpy(b.tab[i:i + bs]).to(DEVICE)
        out.append(torch.sigmoid(model(s, l, t)).cpu().numpy())
    return np.concatenate(out)


def fit_torch(model: nn.Module, reg: Regime, epochs=None, lr=1e-3, bs=512, patience=None):
    epochs, patience = epochs or DL_EPOCHS, patience or DL_PATIENCE
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    tr = reg.fit
    if DL_MAX_TRAIN and len(tr) > DL_MAX_TRAIN:
        tr = tr.take(np.random.RandomState(SEED).permutation(len(tr))[:DL_MAX_TRAIN])
    loader = DataLoader(TensorDataset(tr.seq, tr.lens, torch.from_numpy(tr.tab),
                                      torch.from_numpy(tr.y.astype(np.float32))), batch_size=bs, shuffle=True)
    best, best_state, bad = math.inf, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        for s, l, t, y in loader:
            s, l, t, y = s.to(DEVICE), l.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(s, l, t), y).backward()
            opt.step()
        vloss = log_loss(reg.val.y, np.clip(predict_torch(model, reg.val), 1e-6, 1 - 1e-6))
        hist.append(float(vloss))
        print(f"        epoch {ep + 1}/{epochs}  val loss {vloss:.4f}", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return hist


def torch_trainer(build: Callable[[Regime], nn.Module]):
    def trainer(reg: Regime):
        torch.manual_seed(SEED)
        model = build(reg)
        hist = fit_torch(model, reg)
        fn = _wrap(lambda b: predict_torch(model, b), model)
        fn.history = hist                           # validation loss per epoch (for the training-curve graph)
        return fn
    return trainer


def train_distilbert(reg: Regime, name="distilbert-base-uncased", epochs=2, bs=32):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(DEVICE)

    def enc(urls):
        return tok([canonicalize(u).lower() for u in urls], padding=True, truncation=True, max_length=96,
                   return_tensors="pt")

    n = len(reg.fit) if BERT_MAX_TRAIN is None else min(BERT_MAX_TRAIN, len(reg.fit))
    sel = np.random.RandomState(SEED).permutation(len(reg.fit))[:n]
    urls, y = [reg.fit.urls[i] for i in sel], torch.tensor(reg.fit.y[sel], dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            e = {k: v.to(DEVICE) for k, v in enc([urls[k] for k in j]).items()}
            opt.zero_grad()
            model(**e, labels=y[j].to(DEVICE)).loss.backward()
            opt.step()

    @torch.no_grad()
    def predict(b: Batch):
        model.eval()
        out = []
        for i in range(0, len(b), 128):
            e = {k: v.to(DEVICE) for k, v in enc(b.urls[i:i + 128]).items()}
            out.append(torch.softmax(model(**e).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    return _wrap(predict, model)


TRAINERS: Dict[str, Callable] = {
    "1. Logistic Regression": train_logreg,
    "2. Random Forest": train_rf,
    "3. XGBoost": train_xgb,
    "4. LightGBM": train_lightgbm,
    "5. CatBoost": train_catboost,
    "6. Deep MLP": torch_trainer(lambda r: TabularMLP(len(r.cols))),
    "7. Char-CNN": torch_trainer(lambda r: CharCNN1D()),
    "8. BiLSTM+Attention": torch_trainer(lambda r: BiLSTMAttention()),
    "9. Hybrid CNN-BiLSTM": torch_trainer(lambda r: HybridCNNBiLSTM()),
    "10. DistilBERT": train_distilbert,
    "15. Transformer URL Encoder": torch_trainer(lambda r: TransformerURLEncoder()),
    "16. TabNet-style Gated MLP": torch_trainer(lambda r: TabNetGatedMLP(len(r.cols))),
}
LGBM, CNN, ENS, SUPER_ENS = "4. LightGBM", "7. Char-CNN", "11. Ensemble LGBM+CNN", "11b. Super Ensemble"


# =============================================  =================================
# 5. CALIBRATION + METRICS
# ==============================================================================
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


@dataclass
class ModelOut:
    predict: Callable
    p_cal: np.ndarray
    p_test: np.ndarray
    iso: IsotonicRegression
    platt: LogisticRegression
    thr_tpr95: float
    latency_ms: float


def fit_calibrators(p_cal, y_cal):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, y_cal)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_cal), y_cal)
    thr = float(np.quantile(p_cal[y_cal == 1], 0.05))     # threshold giving ~95% recall on the calibration split
    return iso, platt, thr


def compute_ece(y, p, bins=10):
    edges, ece = np.linspace(0, 1, bins + 1), 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            ece += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(ece)


def compute_mce(y, p, bins=10):
    """Maximum Calibration Error: worst-case bin ECE."""
    edges = np.linspace(0, 1, bins + 1)
    mce = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.any():
            mce = max(mce, abs(p[m].mean() - y[m].mean()))
    return float(mce)


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def precision_at_prevalence(tpr, fpr, prev):
    d = tpr * prev + fpr * (1 - prev)
    return float(tpr * prev / d) if d > 0 else float("nan")


def metrics_row(regime: str, name: str, y, p_raw, iso, platt, thr95, latency_ms=float("nan")) -> dict:
    p_iso, p_pl = iso.predict(p_raw), platt.predict_proba(_logit(p_raw))[:, 1]
    pred = (p_iso >= 0.5).astype(int)
    tn_fp = max(int(((y == 0)).sum()), 1)
    fpr = float(((pred == 1) & (y == 0)).sum() / tn_fp)
    tpr = float(((pred == 1) & (y == 1)).sum() / max(int((y == 1).sum()), 1))
    hit95 = p_raw >= thr95
    fpr95 = float((hit95 & (y == 0)).sum() / tn_fp)
    tpr95 = float((hit95 & (y == 1)).sum() / max(int((y == 1).sum()), 1))
    row = {
        "Regime": regime, "Model": name, "N": len(y), "Phish_share": float(y.mean()),
        "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": _safe(roc_auc_score, y, p_raw), "PR_AUC": _safe(average_precision_score, y, p_raw),
        "Brier_Raw": brier_score_loss(y, p_raw), "Brier_Platt": brier_score_loss(y, p_pl),
        "Brier_Iso": brier_score_loss(y, p_iso),
        "ECE_Raw": compute_ece(y, p_raw), "ECE_Platt": compute_ece(y, p_pl), "ECE_Iso": compute_ece(y, p_iso),
        "MCE_Iso": compute_mce(y, p_iso),
        "FPR@0.5": fpr, "TPR@0.5": tpr, "FPR@TPR95": fpr95, "TPR_at_thr95": tpr95,
        "Latency_ms_single": latency_ms,
    }
    tn, fp, fneg, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    spec = float(tn / max(tn + fp, 1))
    row.update({
        "TN": int(tn), "FP": int(fp), "FN": int(fneg), "TP": int(tp), "Specificity": spec,
        "TPR": tpr, "TNR": spec, "FPR": fpr, "FNR": 1.0 - tpr,
        "Balanced_Acc": (tpr + spec) / 2, "G_Mean": math.sqrt(max(tpr * spec, 0.0)),
        "MCC": _safe(matthews_corrcoef, y, pred), "Kappa": _safe(cohen_kappa_score, y, pred),
        "LogLoss_Iso": _safe(lambda a, b: log_loss(a, np.clip(b, 1e-6, 1 - 1e-6), labels=[0, 1]), y, p_iso),
    })
    for prev in PHISH_PREVALENCES:
        row[f"Precision@{int(prev * 100)}%phish"] = precision_at_prevalence(tpr, fpr, prev)
    return row


DL_PREFIXES = ("6.", "7.", "8.", "9.", "10.", "15.", "16.")


def model_tag(name: str):
    core = re.sub(r"[^A-Za-z0-9]+", "", re.sub(r"^\d+b?\.\s*", "", name))
    if name.startswith(DL_PREFIXES):
        return f"DL_{core}", "Deep Learning"
    if name.startswith("11.") or name.startswith("11b."):
        return f"ENS_{core}", "Ensemble"
    return f"ML_{core}", "Machine Learning"


def print_metric_line(row: dict, indent: str = "    "):
    """One-line summary at the 0.5 threshold on the calibrated probability."""
    print(f"{indent}acc={row['Accuracy']:.4f}  TP={row['TP']}  TN={row['TN']}  FP={row['FP']}  FN={row['FN']}  "
          f"Sens(TPR)={row['TPR']:.4f}  Spec(TNR)={row['TNR']:.4f}  FPR={row['FPR']:.4f}  FNR={row['FNR']:.4f}  "
          f"P={row['Precision']:.4f}  F1={row['F1']:.4f}  AUC={row['ROC_AUC']:.4f}", flush=True)


def print_confusion_summary(df: pd.DataFrame, title: str):
    cols = ["Model", "TP", "TN", "FP", "FN", "TPR", "TNR", "FPR", "FNR", "Accuracy", "Precision", "F1", "ROC_AUC"]
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def single_latency(pred, batch: Batch, n=200) -> float:
    n = min(n, len(batch))
    singles = [batch.take([i]) for i in range(n)]
    pred(singles[0])                                         # warm-up
    t0 = time.perf_counter()
    for s in singles:
        pred(s)
    return (time.perf_counter() - t0) / n * 1000.0


def e2e_latency(pred_fn, urls_sample, cols, scaler, n=100):
    """End-to-end latency: feature extraction + predict for n single URLs.
    Returns (p50_ms, p95_ms)."""
    n = min(n, len(urls_sample))
    lats = []
    # warm-up
    u0 = urls_sample[0]
    d0 = pd.DataFrame([extract_one(u0)])
    pred_fn(make_batch([u0], d0, cols, scaler, encode=True))
    for u in urls_sample[:n]:
        t0 = time.perf_counter()
        df_u = pd.DataFrame([extract_one(u)])
        pred_fn(make_batch([u], df_u, cols, scaler, encode=True))
        lats.append((time.perf_counter() - t0) * 1000.0)
    return float(np.percentile(lats, 50)), float(np.percentile(lats, 95))


def run_models(reg: Regime, names: List[str], measure_latency: bool = True,
               show_metrics: bool = True) -> Dict[str, ModelOut]:
    outs = {}
    for name in names:
        tag, kind = model_tag(name)
        print(f"   Training {tag} ({kind}) on split '{reg.name.lower()}' ...", flush=True)
        pred = TRAINERS[name](reg)
        p_cal, p_test = pred(reg.cal), pred(reg.test)
        iso, platt, thr = fit_calibrators(p_cal, reg.cal.y)
        lat = single_latency(pred, reg.test, n=40 if "DistilBERT" in name else 200) if measure_latency else float("nan")
        outs[name] = ModelOut(pred, p_cal, p_test, iso, platt, thr, lat)
        if show_metrics:
            print_metric_line(metrics_row(reg.name, name, reg.test.y, p_test, iso, platt, thr))
    return outs


def add_ensemble(reg: Regime, outs: Dict[str, ModelOut]):
    if LGBM in outs and CNN in outs:
        a, c = outs[LGBM], outs[CNN]
        pred = _wrap(lambda b: 0.5 * (a.predict(b) + c.predict(b)), None)
        p_cal, p_test = 0.5 * (a.p_cal + c.p_cal), 0.5 * (a.p_test + c.p_test)
        iso, platt, thr = fit_calibrators(p_cal, reg.cal.y)
        outs[ENS] = ModelOut(pred, p_cal, p_test, iso, platt, thr, a.latency_ms + c.latency_ms)
        tag, kind = model_tag(ENS)
        print(f"   Evaluating {tag} ({kind}) on split '{reg.name.lower()}' ...", flush=True)
        print_metric_line(metrics_row(reg.name, ENS, reg.test.y, p_test, iso, platt, thr))
    # Super Ensemble: LightGBM + Char-CNN + Transformer + TabNet-Gated
    _TRANS = "15. Transformer URL Encoder"
    _TABNET = "16. TabNet-style Gated MLP"
    if LGBM in outs and CNN in outs and _TRANS in outs and _TABNET in outs:
        a, c, t, tb = outs[LGBM], outs[CNN], outs[_TRANS], outs[_TABNET]
        def _super_pred(b, _a=a, _c=c, _t=t, _tb=tb):
            return 0.25 * (_a.predict(b) + _c.predict(b) + _t.predict(b) + _tb.predict(b))
        s_pred = _wrap(_super_pred, None)
        s_cal = 0.25 * (a.p_cal + c.p_cal + t.p_cal + tb.p_cal)
        s_test = 0.25 * (a.p_test + c.p_test + t.p_test + tb.p_test)
        iso_s, platt_s, thr_s = fit_calibrators(s_cal, reg.cal.y)
        outs[SUPER_ENS] = ModelOut(s_pred, s_cal, s_test, iso_s, platt_s, thr_s,
                                   a.latency_ms + c.latency_ms + t.latency_ms + tb.latency_ms)
        tag, kind = model_tag(SUPER_ENS)
        print(f"   Evaluating {tag} ({kind}) on split '{reg.name.lower()}' ...", flush=True)
        print_metric_line(metrics_row(reg.name, SUPER_ENS, reg.test.y, s_test, iso_s, platt_s, thr_s))


def evaluate_regime(reg: Regime, outs: Dict[str, ModelOut], mask=None, label=None) -> pd.DataFrame:
    y = reg.test.y
    rows = []
    for name, o in outs.items():
        yy, pp = (y, o.p_test) if mask is None else (y[mask], o.p_test[mask])
        rows.append(metrics_row(label or reg.name, name, yy, pp, o.iso, o.platt, o.thr_tpr95, o.latency_ms))
    return pd.DataFrame(rows)


# ==============================================================================
# 6. TRUST SCORE, RULE OVERLAY, EXPLANATIONS
# ==============================================================================
def trust_band(score: int) -> str:
    return "Trusted" if score >= 80 else ("Caution" if score >= 50 else "High Risk")


def apply_rule_overlay(score: int, f: dict):
    """Transparent caps shown to the user. Evaluate calibration on the *calibrated* score, not on this one."""
    notes = []
    if f["is_ip"]:
        score = min(score, 40)
        notes.append("Address is a raw IP number instead of a registered name.")
    if f["is_shared_host"] and (f["has_intent"] or f["brand_in_sub"] or f["brand_in_label"]):
        score = min(score, 65)
        notes.append("Free shared hosting combined with login/brand wording: score capped.")
    return score, notes


def explain(model, X_row: pd.DataFrame, cols: List[str], top_k=3, thr=0.15) -> List[str]:
    import shap
    sv = shap.TreeExplainer(model).shap_values(X_row[cols])
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., -1]
    contrib = dict(zip(cols, sv[0]))
    gscore = {g: sum(contrib[f] for f in fs if f in contrib) for g, fs in FEATURE_GROUPS.items()}
    risky = sorted([g for g, v in gscore.items() if v > thr], key=lambda g: -gscore[g])[:top_k]
    if risky:
        return [GROUP_RISK_TEXT[g] for g in risky]
    safe = sorted([g for g, v in gscore.items() if v < -thr], key=lambda g: gscore[g])[:top_k]
    return [GROUP_SAFE_TEXT[g] for g in safe] or ["No strong warning signs or strong reassurance found in the address."]


def generate_report_card(url: str, features: dict, shap_reasons: list, trust_score: int, band: str) -> str:
    """Return a formatted multi-line plain-language report card for the given URL."""
    lines = [
        "=" * 70,
        "  WEBSITE TRUST REPORT",
        "=" * 70,
        f"  URL          : {url}",
        f"  Trust score  : {trust_score}/100  [{band}]",
        "",
        "  Risk signals (from SHAP analysis):",
    ]
    for r in shap_reasons:
        lines.append(f"    * {r}")
    lines += [
        "",
        "  Feature values (plain language):",
        f"    is_ip              : {'Yes - raw IP address' if features.get('is_ip') else 'No'}",
        f"    is_shared_host     : {'Yes - shared/free hosting platform' if features.get('is_shared_host') else 'No'}",
        f"    brand_in_sub       : {'Yes - brand name in subdomain' if features.get('brand_in_sub') else 'No'}",
        f"    brand_in_label     : {'Yes - brand name in domain label' if features.get('brand_in_label') else 'No'}",
        f"    intent_count       : {features.get('intent_count', 0)} suspicious keyword(s) found",
        f"    hosting_risk_score : {features.get('hosting_risk_score', 0):.2f} (shared-host x brand/intent)",
        f"    url_len            : {features.get('url_len', 0)} characters",
        f"    is_suspicious_tld  : {'Yes' if features.get('is_suspicious_tld') else 'No'}",
        "",
        "  Footer: Analysis is URL-structure only. No page content was accessed.",
        "=" * 70,
    ]
    return "\n".join(lines)


def score_with_models(url: str, cols, lgbm, cnn, iso_final, iso_lgbm, iso_cnn, scaler, use_ens=True) -> dict:
    f = extract_one(url)
    df = pd.DataFrame([f])
    b = make_batch([url], df, cols, scaler)
    p_l = float(lgbm(b)[0])
    p_c = float(cnn(b)[0])
    p_raw = 0.5 * (p_l + p_c) if use_ens else p_l
    p_cal = float(iso_final.predict([p_raw])[0])
    gap = abs(float(iso_lgbm.predict([p_l])[0]) - float(iso_cnn.predict([p_c])[0]))
    score = int(round(100 * (1 - p_cal)))
    final, notes = apply_rule_overlay(score, f)
    return {"url": url, "calibrated_trust": score, "final_trust": final, "band": trust_band(final),
            "confidence": "Limited information" if gap > DISAGREEMENT_LIMIT else "Normal",
            "model_disagreement": round(gap, 3), "rule_notes": notes, "features": f, "X_row": df}


# ==============================================================================
# 6b. FEATURE SELECTION, DIMENSIONALITY REDUCTION, K-FOLD CV, SHAP, LIME, PLOTS, PRINTING
# ==============================================================================
def _plt():
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")                       # save-only, no windows
    import matplotlib.pyplot as plt
    return plt


def _done(plt):
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def select_features(X_fit: pd.DataFrame, y_fit, cols: List[str], k=None, corr_thr=None) -> pd.DataFrame:
    """Feature selection on the FIT split only (no test leakage):
    1) drop constant features, 2) drop highly correlated duplicates (keep the more informative one),
    3) rank the rest by mutual information, LightGBM gain and L1-logistic |coef|, 4) keep top-k by mean rank."""
    k, corr_thr = k or N_SELECTED_FEATURES, corr_thr or CORR_THRESHOLD
    X = X_fit[cols].reset_index(drop=True)
    y_fit = np.asarray(y_fit)
    sub = np.random.RandomState(SEED).permutation(len(X))[:30000]
    mi = pd.Series(mutual_info_classif(X.iloc[sub], y_fit[sub], random_state=SEED), index=cols)
    corr = X.corr().abs().fillna(0)
    kept, dropped = [], {}
    for c in mi.sort_values(ascending=False).index:
        if X[c].nunique() <= 1:
            dropped[c] = "constant"
            continue
        clash = [kc for kc in kept if corr.loc[c, kc] >= corr_thr]
        if clash:
            dropped[c] = f"corr >= {corr_thr} with {clash[0]}"
        else:
            kept.append(c)
    gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, n_jobs=-1,
                             random_state=SEED, verbose=-1).fit(X[kept], y_fit)
    gain = pd.Series(gbm.booster_.feature_importance(importance_type="gain"), index=kept)
    l1m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED)
    l1 = pd.Series(np.abs(l1m.fit(StandardScaler().fit_transform(X[kept]), y_fit).coef_[0]), index=kept)
    ranks = pd.DataFrame({"mutual_info": mi[kept], "lgbm_gain": gain, "l1_abs_coef": l1}).rank(ascending=False)
    ranks["mean_rank"] = ranks.mean(axis=1)
    tbl = pd.DataFrame(index=cols)
    tbl["mutual_info"], tbl["lgbm_gain"], tbl["l1_abs_coef"] = mi, gain, l1
    tbl["mean_rank"] = ranks["mean_rank"]
    tbl["dropped_reason"] = pd.Series(dropped)
    tbl["selected"] = False
    tbl.loc[ranks["mean_rank"].sort_values().index[:k], "selected"] = True
    tbl = tbl.sort_values("mean_rank").reset_index().rename(columns={"index": "feature"})
    return tbl


def pca_analysis(reg: Regime, plots_dir: str, out_dir: str) -> int:
    """PCA on scaled FIT features: variance table, loadings, scree plot, 2-D PCA and t-SNE maps."""
    plt = _plt()
    pca = PCA(random_state=SEED).fit(reg.fit.tab)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_comp = int(np.searchsorted(cum, PCA_VARIANCE) + 1)
    pd.DataFrame({"component": np.arange(1, len(evr) + 1), "explained_variance": evr,
                  "cumulative": cum}).to_csv(f"{out_dir}/pca_variance.csv", index=False)
    load = pd.DataFrame(pca.components_[:3].T, index=reg.cols, columns=["PC1", "PC2", "PC3"])
    load.to_csv(f"{out_dir}/pca_loadings_top3.csv")
    print(f"      PCA: {n_comp} of {len(reg.cols)} components explain {PCA_VARIANCE:.0%} of the variance "
          f"(PC1 {evr[0]:.1%}, PC2 {evr[1]:.1%}, PC3 {evr[2]:.1%})")

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(reg.fit))[:5000]
    Z, yy = pca.transform(reg.fit.tab[idx]), reg.fit.y[idx]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(range(1, len(evr) + 1), evr, label="per component")
    ax[0].plot(range(1, len(evr) + 1), cum, "r-o", ms=3, label="cumulative")
    ax[0].axhline(PCA_VARIANCE, ls="--", c="gray")
    ax[0].set(xlabel="Principal component", ylabel="Explained variance ratio", title="PCA scree plot")
    ax[0].legend()
    for lab, nm, col in [(0, "Legit", "tab:blue"), (1, "Phishing", "tab:red")]:
        ax[1].scatter(Z[yy == lab, 0], Z[yy == lab, 1], s=4, alpha=0.4, c=col, label=nm)
    ax[1].set(xlabel="PC1", ylabel="PC2", title="PCA projection (5,000 training URLs)")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/pca.png", dpi=130)
    _done(plt)

    if RUN_TSNE:
        from sklearn.manifold import TSNE
        idx2 = rng.permutation(len(reg.fit))[:TSNE_SAMPLES]
        emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED).fit_transform(reg.fit.tab[idx2])
        y2 = reg.fit.y[idx2]
        plt.figure(figsize=(6, 5))
        for lab, nm, col in [(0, "Legit", "tab:blue"), (1, "Phishing", "tab:red")]:
            plt.scatter(emb[y2 == lab, 0], emb[y2 == lab, 1], s=4, alpha=0.5, c=col, label=nm)
        plt.title(f"t-SNE ({TSNE_SAMPLES:,} training URLs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/tsne.png", dpi=130)
        _done(plt)
    return n_comp


# ---- extra trainers: PCA-based and selected-feature models ----
def train_pca_lgbm(reg: Regime):
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED).fit(reg.fit.tab)
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                           n_jobs=-1, random_state=SEED, verbose=-1)
    m.fit(pca.transform(reg.fit.tab), reg.fit.y, eval_set=[(pca.transform(reg.val.tab), reg.val.y)],
          callbacks=[lgb.early_stopping(50, verbose=False)])
    return _wrap(lambda b: m.predict_proba(pca.transform(b.tab))[:, 1], m)


def train_pca_logreg(reg: Regime):
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full", random_state=SEED).fit(reg.fit.tab)
    m = LogisticRegression(max_iter=2000, random_state=SEED).fit(pca.transform(reg.fit.tab), reg.fit.y)
    return _wrap(lambda b: m.predict_proba(pca.transform(b.tab))[:, 1], m)


def train_lgbm_selected(reg: Regime):
    keep = set(SELECTED_COLS) if SELECTED_COLS else set(reg.cols)
    return train_lightgbm(reg, drop=[c for c in reg.cols if c not in keep])


TRAINERS["12. PCA + LightGBM"] = train_pca_lgbm
TRAINERS["13. PCA + LogReg"] = train_pca_logreg
TRAINERS["14. LightGBM (selected features)"] = train_lgbm_selected


# ---- K-fold cross-validation (random stratified AND domain-grouped) ----
def inner_split(train_idx, y, groups, grouped: bool):
    if grouped:
        a, b = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED).split(train_idx, groups=groups[train_idx]))
        fit, hold = train_idx[a], train_idx[b]
        v, c = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(hold, groups=groups[hold]))
    else:
        fit, hold = train_test_split(train_idx, test_size=0.20, stratify=y[train_idx], random_state=SEED)
        v, c = train_test_split(np.arange(len(hold)), test_size=0.50, stratify=y[hold], random_state=SEED)
    return fit, hold[v], hold[c]


CV_KEY_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "MCC", "Balanced_Acc",
                  "Brier_Raw", "Brier_Iso", "ECE_Iso", "FPR@0.5"]


def cross_validate(urls, feat, y, cols, groups, out_dir) -> tuple:
    names = [n for n in CV_MODELS if n in TRAINERS]
    if CV_INCLUDE_DL:
        names += ["7. Char-CNN"]
    idx_all = np.arange(len(y))
    if CV_MAX_ROWS and len(y) > CV_MAX_ROWS:               # subsample whole domains to keep CV fast
        gs = pd.Series(groups)
        sizes = gs.value_counts()
        order = gs.drop_duplicates().values.copy()
        np.random.RandomState(SEED).shuffle(order)
        chosen, total = set(), 0
        for g in order:
            chosen.add(g)
            total += sizes[g]
            if total >= CV_MAX_ROWS:
                break
        idx_all = np.where(gs.isin(chosen).values)[0]
    urls_s = [urls[i] for i in idx_all]
    feat_s = feat.iloc[idx_all].reset_index(drop=True)
    y_s, g_s = y[idx_all], groups[idx_all]
    print(f"      CV data: {len(y_s):,} URLs, phishing share {y_s.mean():.3f}, {CV_FOLDS} folds, models: {names}")

    schemes = [("Random StratifiedKFold", StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED), False),
               ("Grouped StratifiedGroupKFold", StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=SEED), True)]
    frames = []
    for scheme, splitter, grouped in schemes:
        folds = splitter.split(feat_s, y_s, g_s) if grouped else splitter.split(feat_s, y_s)
        for k, (tr, te) in enumerate(folds, 1):
            if grouped:
                assert not (set(g_s[tr]) & set(g_s[te])), "domain leakage between CV train and test"
            fit, val, cal = inner_split(tr, y_s, g_s, grouped)
            reg = build_regime(f"{scheme} fold {k}", urls_s, feat_s, y_s, cols, (fit, val, cal, te),
                               encode=CV_INCLUDE_DL)
            outs = run_models(reg, names, measure_latency=False, show_metrics=False)
            df = evaluate_regime(reg, outs)
            df.insert(0, "Fold", k)
            df.insert(0, "Scheme", scheme)
            frames.append(df)
    cv_all = pd.concat(frames, ignore_index=True)
    cv_all.to_csv(f"{out_dir}/cv_all_folds.csv", index=False)
    g = cv_all.groupby(["Scheme", "Model"])[CV_KEY_METRICS]
    mean, std = g.mean(), g.std()
    mean.add_suffix("_mean").join(std.add_suffix("_std")).to_csv(f"{out_dir}/cv_summary_numeric.csv")
    summ = mean.round(4).astype(str) + " +/- " + std.round(4).astype(str)
    summ.to_csv(f"{out_dir}/cv_summary.csv")
    return summ, cv_all


# ---- SHAP (global) and LIME (local) ----
def _shap_2d(sv):
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    return sv[..., -1] if sv.ndim == 3 else sv


def explain_global(model, cols: List[str], reg: Regime, plots_dir: str, out_dir: str):
    import shap
    plt = _plt()
    X = reg.test.X[cols]
    sample = X.sample(min(SHAP_SAMPLES, len(X)), random_state=SEED)
    sv = _shap_2d(shap.TreeExplainer(model).shap_values(sample))
    imp = pd.Series(np.abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(f"{out_dir}/shap_feature_importance.csv")
    grp = {g: float(np.abs(sv[:, [cols.index(f) for f in fs if f in cols]].sum(axis=1)).mean())
           for g, fs in FEATURE_GROUPS.items() if any(f in cols for f in fs)}
    grp = pd.Series(grp).sort_values(ascending=False)
    grp.rename("mean_abs_shap_group").to_csv(f"{out_dir}/shap_group_importance.csv")
    print("      NOTE: SHAP below explains the LightGBM component only; it is not a full explanation of the LGBM+CNN ensemble.")
    print("      Top 10 features by mean |SHAP|:")
    print(imp.head(10).to_string(float_format=lambda v: f"{v:.4f}"))
    print("      SHAP importance by plain-language group:")
    print(grp.to_string(float_format=lambda v: f"{v:.4f}"))
    gs = grp.sort_values()
    plt.figure(figsize=(8, 4.5))
    plt.barh(np.arange(len(gs)), gs.values, color="tab:orange")
    plt.yticks(np.arange(len(gs)), gs.index)
    plt.xlabel("mean |SHAP| (summed inside the group)")
    plt.title("What drives the score: plain-language feature groups")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/shap_groups.png", dpi=140)
    _done(plt)
    for kind, fname in [("dot", "shap_beeswarm.png"), ("bar", "shap_bar.png")]:
        shap.summary_plot(sv, sample, plot_type=kind, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/{fname}", dpi=130)
        _done(plt)
    return sv, sample


def run_lime(model, cols: List[str], reg: Regime, sv, sample: pd.DataFrame, plots_dir: str):
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print("      LIME not installed (pip install lime) - skipped.")
        return
    train_np = reg.fit.X[cols].sample(min(20000, len(reg.fit)), random_state=SEED).values
    expl = LimeTabularExplainer(train_np, feature_names=cols, class_names=["Legit", "Phishing"],
                                mode="classification", discretize_continuous=True, random_state=SEED,
                                feature_selection="highest_weights")
    predict = lambda a: model.predict_proba(pd.DataFrame(a, columns=cols))
    ytest, Xtest = reg.test.y, reg.test.X[cols]
    picks = {"phishing_example": int(np.where(ytest == 1)[0][0]), "legit_example": int(np.where(ytest == 0)[0][0])}
    shared = np.where(reg.test.X["is_shared_host"].values == 1)[0]
    if len(shared):
        picks["shared_host_example"] = int(shared[0])
    for tag, i in picks.items():
        exp = expl.explain_instance(Xtest.iloc[i].values, predict, num_features=8, num_samples=2000)
        exp.save_to_file(f"{plots_dir}/lime_{tag}.html")
        print(f"      LIME [{tag}] {reg.test.urls[i][:70]}")
        for desc, w in exp.as_list():
            print(f"         {w:+.3f}  {desc}")
    overlaps = []
    for i in range(min(LIME_SAMPLES, len(sample))):
        exp = expl.explain_instance(sample.iloc[i].values, predict, num_features=5, num_samples=1000)
        lime_top = {cols[j] for j, _ in exp.as_map()[1]}
        shap_top = set(np.array(cols)[np.argsort(-np.abs(sv[i]))[:5]])
        overlaps.append(len(lime_top & shap_top) / 5)
    print(f"      SHAP vs LIME top-5 feature agreement over {len(overlaps)} URLs: {np.mean(overlaps):.2f} "
          f"(1.00 = identical)")


# ---- curves ----
def make_plots(reg: Regime, outs: Dict[str, "ModelOut"], plots_dir: str):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve
    plt = _plt()
    y = reg.test.y
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, o in outs.items():
        fpr, tpr, _ = roc_curve(y, o.p_test)
        pr, rc, _ = precision_recall_curve(y, o.p_test)
        ax[0].plot(fpr, tpr, label=name, lw=1)
        ax[1].plot(rc, pr, label=name, lw=1)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.6)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (domain-disjoint test)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall (domain-disjoint test)")
    ax[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/roc_pr.png", dpi=130)
    _done(plt)

    fig, axs = plt.subplots(1, min(3, len(outs)), figsize=(5 * min(3, len(outs)), 4.5), squeeze=False)
    for a, name in zip(axs[0], [n for n in (LGBM, CNN, ENS) if n in outs] or list(outs)[:3]):
        o = outs[name]
        for lab, p in [("raw", o.p_test), ("isotonic", o.iso.predict(o.p_test))]:
            fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
            a.plot(mp, fr, "o-", ms=3, label=lab)
        a.plot([0, 1], [0, 1], "k--", lw=0.6)
        a.set(title=f"Reliability: {name}", xlabel="Predicted P(phishing)", ylabel="Observed frequency")
        a.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/reliability.png", dpi=130)
    _done(plt)


# ---- printing every metric ----
COUNT_ROWS = {"N", "TN", "FP", "FN", "TP"}


def _fmt(v, count=False):
    if pd.isna(v):
        return "nan"
    return str(int(v)) if count else f"{v:.4f}"


def print_all_metrics(df: pd.DataFrame, title: str, chunk: int = 4):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)
    t = df.set_index("Model").drop(columns=["Regime"], errors="ignore").T
    t = t.apply(lambda row: row.map(lambda v: _fmt(v, row.name in COUNT_ROWS)), axis=1)
    for i in range(0, t.shape[1], chunk):
        print(t.iloc[:, i:i + chunk].to_string())
        print()


def print_best_report(reg: Regime, outs: Dict[str, "ModelOut"], res: pd.DataFrame):
    best = res.loc[res["PR_AUC"].idxmax(), "Model"]
    o = outs[best]
    pred = (o.iso.predict(o.p_test) >= 0.5).astype(int)
    print("\n" + "=" * 100 + f"\nClassification report + confusion matrix: best PR-AUC model = {best}\n" + "=" * 100)
    print(classification_report(reg.test.y, pred, target_names=["Legit", "Phishing"], digits=4))
    cm = confusion_matrix(reg.test.y, pred, labels=[0, 1])
    print(pd.DataFrame(cm, index=["Actual Legit", "Actual Phishing"], columns=["Pred Legit", "Pred Phishing"]))


# ==============================================================================
# 6c. COMPARATIVE + METRIC GRAPHS  (every figure is saved to <out>/plots/ and, if SHOW_PLOTS, also displayed)
# ==============================================================================
BAR_METRICS = ("Accuracy", "Precision", "TPR", "TNR", "F1", "ROC_AUC", "PR_AUC")
HEAT_METRICS = [("Accuracy", 1), ("Precision", 1), ("TPR", 1), ("TNR", 1), ("F1", 1), ("ROC_AUC", 1), ("PR_AUC", 1),
                ("MCC", 1), ("Balanced_Acc", 1), ("Precision@5%phish", 1), ("Precision@1%phish", 1),
                ("FPR", -1), ("FNR", -1), ("Brier_Iso", -1), ("ECE_Iso", -1), ("LogLoss_Iso", -1)]


def _short(name: str) -> str:
    return re.sub(r"^\d+b?\.\s*", "", name)


def _lab(names, width: int = 14):
    return [textwrap.fill(_short(s), width) for s in names]


def _uniq(res: pd.DataFrame) -> pd.DataFrame:
    return res.drop_duplicates("Model").reset_index(drop=True)


def _save(plt, plots_dir: str, fname: str):
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{fname}", dpi=140, bbox_inches="tight")
    _done(plt)


def plot_metric_bars(res, plots_dir, title, fname, metrics=BAR_METRICS):
    """Grouped bars: every model, several metrics side by side."""
    plt = _plt()
    df = _uniq(res)
    n, m = len(df), len(metrics)
    w, x = 0.8 / m, np.arange(n)
    fig, ax = plt.subplots(figsize=(max(11, 1.0 * n + 3), 6.5))
    for i, met in enumerate(metrics):
        ax.bar(x + (i - (m - 1) / 2) * w, df[met].values, w, label=met)
    ax.set_xticks(x)
    ax.set_xticklabels(_lab(df["Model"]), rotation=40, ha="right", fontsize=8)
    ax.set_ylim(max(0.0, float(np.nanmin(df[list(metrics)].values)) - 0.05), 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title, pad=34)
    ax.legend(ncol=m, loc="upper center", bbox_to_anchor=(0.5, 1.10), fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(plt, plots_dir, fname)


def plot_rankings(res, plots_dir, fname="cmp_rankings.png", metrics=("Accuracy", "F1", "PR_AUC", "MCC")):
    """Sorted bars per metric; the best model is green (high-accuracy view)."""
    plt = _plt()
    df = _uniq(res)
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for ax, met in zip(axs.ravel(), metrics):
        d = df.sort_values(met).reset_index(drop=True)
        best = d[met].max()
        ax.barh(np.arange(len(d)), d[met], color=["tab:green" if v == best else "tab:blue" for v in d[met]])
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(_lab(d["Model"], 30), fontsize=8)
        for yi, v in enumerate(d[met]):
            ax.text(v, yi, f" {v:.4f}", va="center", fontsize=7)
        ax.set_xlim(max(0.0, d[met].min() - 0.04), min(1.05, d[met].max() + 0.05))
        ax.set_title(f"{met} (higher is better, best in green)")
        ax.grid(axis="x", alpha=0.3)
    _save(plt, plots_dir, fname)


def plot_tpr_tnr(res, plots_dir, fname="tpr_tnr_comparison.png"):
    """TPR vs TNR bars, ROC-style operating points (FPR vs TPR) and the two error rates (FPR, FNR)."""
    plt = _plt()
    df = _uniq(res)
    x, w = np.arange(len(df)), 0.38
    fig, ax = plt.subplots(1, 3, figsize=(max(21, 1.3 * len(df) + 8), 6.8))
    ax[0].bar(x - w / 2, df["TPR"], w, color="tab:green", label="TPR / sensitivity (phishing caught)")
    ax[0].bar(x + w / 2, df["TNR"], w, color="tab:blue", label="TNR / specificity (legit allowed)")
    ax[0].set_ylim(max(0.0, float(np.nanmin(df[["TPR", "TNR"]].values)) - 0.05), 1.0)
    ax[0].set_title("TPR vs TNR (higher is better)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(_lab(df["Model"]), rotation=40, ha="right", fontsize=8)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].grid(axis="y", alpha=0.3)
    ax[1].scatter(df["FPR"], df["TPR"], s=70, c="tab:purple", zorder=3)
    for _, r in df.iterrows():
        ax[1].annotate(_short(r["Model"]), (r["FPR"], r["TPR"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    pad_x = 0.15 * (df["FPR"].max() - df["FPR"].min() + 1e-6)
    pad_y = 0.15 * (df["TPR"].max() - df["TPR"].min() + 1e-6)
    ax[1].set_xlim(max(0, df["FPR"].min() - pad_x), df["FPR"].max() + pad_x)
    ax[1].set_ylim(df["TPR"].min() - pad_y, min(1.0, df["TPR"].max() + pad_y))
    ax[1].set(xlabel="FPR = 1 - TNR (false alarms, lower is better)", ylabel="TPR (higher is better)",
              title="Operating point at threshold 0.5 (top-left is best)")
    ax[1].grid(alpha=0.3)
    ax[2].bar(x - w / 2, df["FPR"], w, color="tab:orange", label="FPR (legit flagged)")
    ax[2].bar(x + w / 2, df["FNR"], w, color="tab:red", label="FNR (phishing missed)")
    ax[2].set_title("Error rates (lower is better)")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(_lab(df["Model"]), rotation=40, ha="right", fontsize=8)
    ax[2].legend(fontsize=8)
    ax[2].grid(axis="y", alpha=0.3)
    _save(plt, plots_dir, fname)


def plot_heatmap(res, plots_dir, fname="cmp_heatmap_all_metrics.png", title="All metrics, all models"):
    """Heat map of every metric. Colour = rank within the metric (green best); text = raw value."""
    plt = _plt()
    df = _uniq(res)
    mets = [(m, d) for m, d in HEAT_METRICS if m in df.columns]
    raw = df[[m for m, _ in mets]].values.astype(float)
    score = np.zeros_like(raw)
    for j, (_, d) in enumerate(mets):
        col = raw[:, j]
        rng = np.nanmax(col) - np.nanmin(col)
        s = (col - np.nanmin(col)) / rng if rng > 0 else np.full_like(col, 0.5)
        score[:, j] = s if d == 1 else 1 - s
    fig, ax = plt.subplots(figsize=(len(mets) * 0.95 + 4, len(df) * 0.55 + 2.5))
    ax.imshow(score, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([m + (" (low=best)" if d == -1 else "") for m, d in mets], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([_short(s) for s in df["Model"]], fontsize=8)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            dark = score[i, j] < 0.12 or score[i, j] > 0.88          # white text on very dark cells
            ax.text(j, i, "nan" if np.isnan(raw[i, j]) else f"{raw[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if dark else "black")
    ax.set_title(title + "  (green = best in that column)")
    _save(plt, plots_dir, fname)


def plot_confusion_grid(res, plots_dir, fname="cmp_confusion_matrices.png"):
    """One confusion matrix per model (counts + row-normalised %)."""
    plt = _plt()
    df = _uniq(res)
    ncol = 4
    nrow = math.ceil(len(df) / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axs.ravel(), df.iterrows()):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.axis("on")
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred Legit", "Pred Phish"], fontsize=8)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Legit", "Phish"], fontsize=8)
        ax.set_title(textwrap.fill(_short(r["Model"]), 30), fontsize=9)
    plt.suptitle("Confusion matrices (threshold 0.5 on calibrated probability)", y=1.0)
    _save(plt, plots_dir, fname)


def plot_error_profile(res, plots_dir, fname="cmp_errors_tpr_tnr.png"):
    """Left: FP and FN counts.  Right: TPR (sensitivity) vs TNR (specificity)."""
    plt = _plt()
    df = _uniq(res)
    x, w = np.arange(len(df)), 0.38
    fig, ax = plt.subplots(1, 2, figsize=(max(15, 1.2 * len(df) + 5), 6.5))
    ax[0].bar(x - w / 2, df["FP"], w, color="tab:orange", label="FP: legit flagged as phishing")
    ax[0].bar(x + w / 2, df["FN"], w, color="tab:red", label="FN: phishing missed")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Error counts (lower is better)")
    ax[1].bar(x - w / 2, df["TPR"], w, color="tab:green", label="TPR (sensitivity)")
    ax[1].bar(x + w / 2, df["TNR"], w, color="tab:blue", label="TNR (specificity)")
    ax[1].set_ylim(max(0.0, float(np.nanmin(df[["TPR", "TNR"]].values)) - 0.05), 1.0)
    ax[1].set_title("TPR vs TNR (higher is better)")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(_lab(df["Model"]), rotation=40, ha="right", fontsize=8)
        a.legend(fontsize=8)
        a.grid(axis="y", alpha=0.3)
    _save(plt, plots_dir, fname)


def plot_calibration_bars(res, plots_dir, fname="cmp_calibration.png"):
    """ECE / MCE and Brier score before and after calibration (Gap 2)."""
    plt = _plt()
    df = _uniq(res)
    x = np.arange(len(df))
    # ECE panel: 4 bars (ECE_Raw, ECE_Platt, ECE_Iso, MCE_Iso)
    ece_keys = ["ECE_Raw", "ECE_Platt", "ECE_Iso"]
    if "MCE_Iso" in df.columns:
        ece_keys.append("MCE_Iso")
    brier_keys = ["Brier_Raw", "Brier_Platt", "Brier_Iso"]
    n_ece = len(ece_keys)
    w_ece = 0.7 / n_ece
    w_brier = 0.27
    fig, ax = plt.subplots(1, 2, figsize=(max(15, 1.2 * len(df) + 5), 6))
    for i, k in enumerate(ece_keys):
        offsets = x + (i - (n_ece - 1) / 2) * w_ece
        ax[0].bar(offsets, df[k] if k in df.columns else 0, w_ece,
                  label=k.replace("_", " "))
    ax[0].set_title("ECE & MCE (lower is better)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(_lab(df["Model"]), rotation=40, ha="right", fontsize=8)
    ax[0].legend(title="calibration", fontsize=7)
    ax[0].grid(axis="y", alpha=0.3)
    for i, k in enumerate(brier_keys):
        ax[1].bar(x + (i - 1) * w_brier, df[k], w_brier, label=k.split("_")[1])
    ax[1].set_title("Brier score (lower is better)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(_lab(df["Model"]), rotation=40, ha="right", fontsize=8)
    ax[1].legend(title="calibration")
    ax[1].grid(axis="y", alpha=0.3)
    _save(plt, plots_dir, fname)


def plot_deployment_precision(res, plots_dir, fname="cmp_realworld_precision.png"):
    """Precision at realistic phishing prevalence, and false-alarm rate needed for 95% recall."""
    plt = _plt()
    df = _uniq(res)
    x, w = np.arange(len(df)), 0.4
    fig, ax = plt.subplots(1, 2, figsize=(max(15, 1.2 * len(df) + 5), 6))
    ax[0].bar(x - w / 2, df["Precision@5%phish"], w, label="5% of traffic is phishing")
    ax[0].bar(x + w / 2, df["Precision@1%phish"], w, label="1% of traffic is phishing")
    ax[0].set_title("Expected precision of alerts in real traffic (higher is better)")
    ax[1].bar(x, df["FPR@TPR95"], 0.6, color="tab:purple")
    ax[1].set_title("False-alarm rate needed to catch 95% of phishing (lower is better)")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(_lab(df["Model"]), rotation=40, ha="right", fontsize=8)
        a.grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8)
    _save(plt, plots_dir, fname)


def plot_latency_tradeoff(res, plots_dir, fname="cmp_latency_accuracy.png"):
    """Model latency per URL (log scale) and the accuracy-vs-speed trade-off (Gap 4)."""
    plt = _plt()
    df = _uniq(res).dropna(subset=["Latency_ms_single"]).sort_values("Latency_ms_single").reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    ax[0].barh(np.arange(len(df)), df["Latency_ms_single"], color="tab:cyan")
    ax[0].set_yticks(np.arange(len(df)))
    ax[0].set_yticklabels(_lab(df["Model"], 30), fontsize=8)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("milliseconds per URL (model only, log scale)")
    ax[0].set_title("Inference latency (lower is better)")
    for yi, v in enumerate(df["Latency_ms_single"]):
        ax[0].text(v, yi, f" {v:.2f}", va="center", fontsize=7)
    ax[1].scatter(df["Latency_ms_single"], df["Accuracy"], s=60, c="tab:blue")
    for _, r in df.iterrows():
        ax[1].annotate(_short(r["Model"]), (r["Latency_ms_single"], r["Accuracy"]), fontsize=7,
                       xytext=(4, 4), textcoords="offset points")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="milliseconds per URL (log scale)", ylabel="Accuracy",
              title="Accuracy vs latency (top-left is best)")
    ax[1].grid(alpha=0.3)
    _save(plt, plots_dir, fname)


def plot_random_vs_domain(res_a, res_b, plots_dir, fname="gap1_random_vs_unseen.png",
                          metrics=("Accuracy", "TPR", "TNR", "F1")):
    """Gap 1: random split vs domain-disjoint split."""
    plt = _plt()
    m = _uniq(res_a).merge(_uniq(res_b), on="Model", suffixes=("_r", "_d"))
    x, w = np.arange(len(m)), 0.38
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for ax, met in zip(axs.ravel(), metrics):
        ax.bar(x - w / 2, m[f"{met}_r"], w, label="random split (inflated)", color="tab:gray")
        ax.bar(x + w / 2, m[f"{met}_d"], w, label="unseen domains", color="tab:red")
        ax.set_ylim(max(0.0, float(min(m[f"{met}_r"].min(), m[f"{met}_d"].min())) - 0.03), 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(_lab(m["Model"]), rotation=30, ha="right", fontsize=8)
        ax.set_title(met)
        ax.grid(axis="y", alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    plt.suptitle("Gap 1: random split vs domain-disjoint split")
    _save(plt, plots_dir, fname)


def plot_cv(cv_all, plots_dir, fname="cv_random_vs_grouped.png", metrics=("Accuracy", "TPR", "TNR", "PR_AUC")):
    """K-fold CV: mean +/- std per model, one bar per CV scheme, dots = individual folds."""
    plt = _plt()
    schemes, models = list(cv_all["Scheme"].unique()), list(cv_all["Model"].unique())
    w = 0.8 / len(schemes)
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    for ax, met in zip(axs.ravel(), metrics):
        for si, sch in enumerate(schemes):
            sub = cv_all[cv_all["Scheme"] == sch]
            xs = np.arange(len(models)) + (si - (len(schemes) - 1) / 2) * w
            means = [sub[sub["Model"] == mm][met].mean() for mm in models]
            stds = [sub[sub["Model"] == mm][met].std() for mm in models]
            ax.bar(xs, means, w, yerr=stds, capsize=3, label=sch)
            for xi, mm in zip(xs, models):
                pts = sub[sub["Model"] == mm][met].values
                ax.scatter(np.full(len(pts), xi), pts, s=10, c="k", alpha=0.6, zorder=3)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(_lab(models, 16), fontsize=8)
        ax.set_ylim(cv_all[met].min() - 0.02, min(1.0, cv_all[met].max() + 0.015))
        ax.set_title(f"{met}: mean +/- std over folds")
        ax.grid(axis="y", alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    plt.suptitle(f"{CV_FOLDS}-fold cross-validation: random vs domain-grouped folds")
    _save(plt, plots_dir, fname)


def plot_slice(slice_tbl, plots_dir, fname="gap3_shared_hosting_slice.png"):
    """Gap 3: performance on URLs hosted on free shared platforms."""
    plt = _plt()
    df = _uniq(slice_tbl)
    n, legit = int(df["N"].iloc[0]), int(round(df["N"].iloc[0] * (1 - df["Phish_share"].iloc[0])))
    x, w = np.arange(len(df)), 0.27
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, (k, c) in enumerate([("TPR", "tab:green"), ("TNR", "tab:blue"), ("F1", "tab:purple")]):
        ax.bar(x + (i - 1) * w, df[k], w, label=k, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(_lab(df["Model"], 18), fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Shared-hosting slice: N={n:,} URLs but only {legit} legitimate -> TNR is unreliable")
    _save(plt, plots_dir, fname)


def plot_feature_selection(fs_tbl, plots_dir, fname="feature_selection.png"):
    """Feature selection ranking: mutual information, LightGBM gain, L1 coefficient."""
    plt = _plt()
    d = fs_tbl.sort_values("mean_rank", na_position="last").reset_index(drop=True).iloc[::-1].reset_index(drop=True)
    status = np.where(d["selected"], "selected", np.where(d["dropped_reason"].notna(), "dropped (redundant)", "not selected"))
    cmap = {"selected": "tab:green", "not selected": "tab:gray", "dropped (redundant)": "tab:red"}
    colors = [cmap[s] for s in status]
    fig, axs = plt.subplots(1, 3, figsize=(17, max(7, 0.28 * len(d))), sharey=True)
    for ax, (col, title) in zip(axs, [("mutual_info", "Mutual information"), ("lgbm_gain", "LightGBM gain (symlog)"),
                                      ("l1_abs_coef", "L1-logistic |coef|")]):
        ax.barh(np.arange(len(d)), d[col].fillna(0), color=colors)
        ax.set_title(title)
        if col == "lgbm_gain":
            ax.set_xscale("symlog")
    axs[0].set_yticks(np.arange(len(d)))
    axs[0].set_yticklabels(d["feature"], fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.values()]
    axs[2].legend(handles, cmap.keys(), loc="lower right", fontsize=8)
    plt.suptitle("Feature selection (best average rank at top)")
    _save(plt, plots_dir, fname)


def plot_training_curves(outs, plots_dir, fname="dl_training_curves.png"):
    """Validation loss per epoch for the deep models."""
    plt = _plt()
    hist = {n: o.predict.history for n, o in outs.items() if getattr(o.predict, "history", None)}
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, h in hist.items():
        ax.plot(range(1, len(h) + 1), h, "o-", label=_short(n))
    ax.set(xlabel="Epoch", ylabel="Validation log-loss", title="Deep model training curves (lower is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(plt, plots_dir, fname)


def plot_trust_distribution(reg, outs, key, plots_dir, fname="trust_score_distribution.png"):
    """Trust score (0-100) distribution for truly legitimate vs truly phishing test URLs."""
    plt = _plt()
    o = outs[key]
    trust = 100.0 * (1.0 - o.iso.predict(o.p_test))
    y = reg.test.y
    bins = np.arange(0, 105, 5)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(trust[y == 0], bins=bins, alpha=0.6, color="tab:blue", label="Legitimate sites")
    ax.hist(trust[y == 1], bins=bins, alpha=0.6, color="tab:red", label="Phishing sites")
    ax.axvline(50, ls="--", c="k", lw=0.8)
    ax.axvline(80, ls="--", c="k", lw=0.8)
    ymax = ax.get_ylim()[1]
    for xv, t in [(25, "High Risk"), (65, "Caution"), (90, "Trusted")]:
        ax.text(xv, ymax * 0.95, t, ha="center", fontsize=9)
    ax.set(xlabel="Trust score (100 = most trusted)", ylabel="Number of test URLs",
           title=f"Trust score distribution on unseen domains: {_short(key)}")
    ax.legend(loc="center left")
    _save(plt, plots_dir, fname)


def plot_radar(res, plots_dir, fname="cmp_radar_top_models.png", top=4,
               metrics=("Accuracy", "Precision", "TPR", "TNR", "F1", "ROC_AUC")):
    """Radar chart of the top models by PR-AUC."""
    plt = _plt()
    df = _uniq(res).nlargest(top, "PR_AUC")
    ang = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, polar=True)
    lo = float(df[list(metrics)].values.min()) - 0.05
    for _, r in df.iterrows():
        v = [r[m] for m in metrics]
        v += v[:1]
        ax.plot(ang, v, lw=1.8, label=_short(r["Model"]))
        ax.fill(ang, v, alpha=0.08)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(lo, 1.0)
    ax.set_title(f"Metric profile of the top {top} models (by PR-AUC)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    _save(plt, plots_dir, fname)


def make_all_graphs(res_b, res_a, slice_tbl, res_x, cv_all, fs_tbl, outs_b, reg_b, plots_dir, final_key):
    jobs = [
        ("bars: all metrics (unseen domains)", lambda: plot_metric_bars(
            res_b, plots_dir, "Model comparison on UNSEEN domains", "cmp_metrics_unseen_domains.png")),
        ("TPR / TNR comparison", lambda: plot_tpr_tnr(res_b, plots_dir)),
        ("rankings: accuracy / F1 / PR-AUC / MCC", lambda: plot_rankings(res_b, plots_dir)),
        ("heat map: every metric", lambda: plot_heatmap(res_b, plots_dir)),
        ("radar: top models", lambda: plot_radar(res_b, plots_dir)),
        ("confusion matrices", lambda: plot_confusion_grid(res_b, plots_dir)),
        ("FP/FN and TPR/TNR", lambda: plot_error_profile(res_b, plots_dir)),
        ("calibration (ECE, Brier)", lambda: plot_calibration_bars(res_b, plots_dir)),
        ("real-world precision", lambda: plot_deployment_precision(res_b, plots_dir)),
        ("latency trade-off", lambda: plot_latency_tradeoff(res_b, plots_dir)),
        ("random vs unseen domains", lambda: plot_random_vs_domain(res_a, res_b, plots_dir)),
        ("feature selection", lambda: plot_feature_selection(fs_tbl, plots_dir)),
        ("DL training curves", lambda: plot_training_curves(outs_b, plots_dir)),
        ("trust score distribution", lambda: plot_trust_distribution(reg_b, outs_b, final_key, plots_dir)),
    ]
    if len(res_a):
        jobs.append(("random-split metrics", lambda: plot_metric_bars(
            res_a, plots_dir, "Model comparison on RANDOM split (inflated)", "cmp_metrics_random_split.png")))
    if len(cv_all):
        jobs.append(("cross-validation", lambda: plot_cv(cv_all, plots_dir)))
    if len(slice_tbl):
        jobs.append(("shared-hosting slice", lambda: plot_slice(slice_tbl, plots_dir)))
    if len(res_x):
        jobs.append(("cross-dataset", lambda: plot_metric_bars(
            res_x, plots_dir, "Model comparison on the EXTERNAL dataset", "cmp_metrics_cross_dataset.png")))
        jobs.append(("cross-dataset TPR/TNR", lambda: plot_tpr_tnr(res_x, plots_dir, "tpr_tnr_cross_dataset.png")))
    for name, fn in jobs:
        try:
            fn()
        except Exception as e:                      # one failed graph must never stop the pipeline
            print(f"      [graph skipped] {name}: {type(e).__name__}: {e}")
    print(f"      {len([f for f in os.listdir(plots_dir) if f.endswith('.png')])} PNG graphs in {plots_dir}")


# ==============================================================================
# 7. TRAIN COMMAND
# ==============================================================================
def load_dataset(path: str):
    df = pd.read_csv(path)
    missing = {"URL", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df["label"].isna().any():
        raise ValueError("Dataset label column contains missing values.")
    observed = set(pd.to_numeric(df["label"], errors="coerce").dropna().unique())
    if observed != {0, 1}:
        raise ValueError(f"Expected binary PhiUSIIL labels 0 and 1, found: {sorted(observed)}")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["_canon"] = df["URL"].map(lambda u: canonicalize(u).lower())
    df = df.drop_duplicates(subset="_canon").reset_index(drop=True)
    if QUICK_TEST_ROWS:
        df = df.sample(min(QUICK_TEST_ROWS, len(df)), random_state=SEED).reset_index(drop=True)
    y = (df["label"] == 0).astype(int).values            # PhiUSIIL: 1 = legit, 0 = phishing  ->  1 = phishing
    return df["URL"].tolist(), y


def cmd_train(args):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    out, plots = args.out, f"{args.out}/plots"
    os.makedirs(plots, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    ff = lambda v: f"{v:.4f}"

    print(f"[1/11] Loading {args.data} (device: {DEVICE})")
    urls, y = load_dataset(args.data)
    print(f"       {len(urls):,} unique URLs | phishing share {y.mean():.3f}")

    print("[2/11] Extracting features (PSL-aware, tenant-level) ...")
    feat = extract_features(urls)
    all_cols = numeric_columns(feat)
    check_groups(all_cols)
    groups = feat["group_domain"].values
    print(f"       {len(all_cols)} numeric features | {len(set(groups)):,} groups | "
          f"shared-host rows: {int(feat['is_shared_host'].sum()):,}")

    print("[3/11] Feature selection (fit split only: constant -> correlation -> MI / LightGBM gain / L1 ranking) ...")
    split_b = grouped_split(groups, SEED)
    fs_tbl = select_features(feat.iloc[split_b[0]], y[split_b[0]], all_cols)
    SELECTED_COLS[:] = fs_tbl.loc[fs_tbl["selected"], "feature"].tolist()
    fs_tbl.to_csv(f"{out}/feature_selection.csv", index=False)
    print(fs_tbl.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"       Selected {len(SELECTED_COLS)}/{len(all_cols)}: {SELECTED_COLS}")
    cols = list(SELECTED_COLS) if USE_SELECTED_FEATURES else list(all_cols)

    all_names = [n for n in TRAINERS if (RUN_DISTILBERT or "DistilBERT" not in n) and n not in SKIP_MODELS]

    print("[4/11] Regime B: domain-disjoint split (primary) + dimensionality reduction ...")
    reg_b = build_regime("Domain-disjoint", urls, feat, y, cols, split_b)
    print(f"       fit {len(reg_b.fit):,} | val {len(reg_b.val):,} | cal {len(reg_b.cal):,} | test {len(reg_b.test):,}")
    pca_analysis(reg_b, plots, out)
    outs_b = run_models(reg_b, all_names)
    add_ensemble(reg_b, outs_b)
    res_b = evaluate_regime(reg_b, outs_b)

    print("[5/11] Regime A: random split (what most papers report) ...")
    reg_a = build_regime("Random", urls, feat, y, cols, random_split(y, SEED))
    outs_a = run_models(reg_a, [n for n in all_names if n in RANDOM_REGIME_MODELS])
    res_a = evaluate_regime(reg_a, outs_a)
    gap = res_a.merge(res_b, on="Model", suffixes=("_random", "_domain"))
    gap_tbl = pd.DataFrame({
        "Model": gap["Model"],
        "F1_random": gap["F1_random"], "F1_domain": gap["F1_domain"], "F1_drop": gap["F1_random"] - gap["F1_domain"],
        "PR_AUC_random": gap["PR_AUC_random"], "PR_AUC_domain": gap["PR_AUC_domain"],
        "PR_AUC_drop": gap["PR_AUC_random"] - gap["PR_AUC_domain"],
    })

    print("[6/11] Gap 3: shared-hosting slice + hosting-feature ablation ...")
    mask = reg_b.test.X["is_shared_host"].values == 1
    ym = reg_b.test.y[mask]
    shared_legit, shared_phish = int((ym == 0).sum()), int((ym == 1).sum())
    print(f"       slice size {int(mask.sum()):,} (phish {shared_phish:,}, legit {shared_legit:,})")
    if shared_legit < 100:
        print("       WARNING: very few legitimate shared-host examples; accuracy/PR-AUC on this slice are prevalence-dominated.")
        print("       Treat shared-host specificity (TNR) and false-positive rate as unstable until more legitimate tenants are collected.")
    slice_tbl = pd.DataFrame()
    if mask.sum() >= 30 and 0 < ym.sum() < len(ym):
        abl_pred = train_lightgbm(reg_b, drop=HOSTING_FEATURES)
        pc, pt = abl_pred(reg_b.cal), abl_pred(reg_b.test)
        iso, platt, thr = fit_calibrators(pc, reg_b.cal.y)
        abl = ModelOut(abl_pred, pc, pt, iso, platt, thr, float("nan"))
        slice_outs = {k: outs_b[k] for k in (LGBM, CNN, ENS) if k in outs_b}
        slice_outs["4b. LightGBM WITHOUT hosting features"] = abl
        slice_tbl = evaluate_regime(reg_b, slice_outs, mask=mask, label="Shared-host slice")
        full_abl = evaluate_regime(reg_b, {"4b. LightGBM WITHOUT hosting features": abl}, label="Domain-disjoint")
        res_b = pd.concat([res_b, full_abl], ignore_index=True)
    else:
        print("       Slice too small / single-class: collect more *.vercel.app, *.netlify.app, ... tenant URLs.")

    # Gap 3 extra: top-5 most common eTLD+1 suffixes in the shared-host test slice
    if mask.sum() >= 1:
        shared_domains = reg_b.test.X["group_domain"].values[mask]
        # extract suffix part (after first dot, or the domain itself)
        def _etld1_suffix(d):
            parts = str(d).split(".")
            return ".".join(parts[1:]) if len(parts) > 1 else d
        suffix_counts = Counter([_etld1_suffix(d) for d in shared_domains])
        top5 = suffix_counts.most_common(5)
        print("       Top-5 eTLD+1 suffixes in shared-host test slice:")
        for sfx, cnt in top5:
            print(f"         {sfx:40s}  {cnt:4d} URLs")

    print("[7/11] Cross-dataset regime ...")
    res_x = pd.DataFrame()
    if args.external and os.path.exists(args.external):
        ext = pd.read_csv(args.external)
        ext_urls, ext_y = ext["url"].astype(str).tolist(), ext["label"].astype(int).values
        ext_batch = make_batch(ext_urls, extract_features(ext_urls), cols, reg_b.scaler, ext_y)
        rows_x = []
        for n, o in outs_b.items():
            tag, kind = model_tag(n)
            print(f"   Evaluating {tag} ({kind}) on split 'cross-dataset' ...", flush=True)
            r = metrics_row("Cross-dataset", n, ext_y, o.predict(ext_batch), o.iso, o.platt, o.thr_tpr95)
            print_metric_line(r)
            rows_x.append(r)
        res_x = pd.DataFrame(rows_x)
    else:
        print("       (skipped: set EXTERNAL_FILE to a CSV with columns url,label; 1 = phishing)")

    print("[8/11] Gap 4: end-to-end latency (feature extraction + model) for ALL models ...")
    lat = []
    for u in reg_b.test.urls[:500]:
        t0 = time.perf_counter()
        d = pd.DataFrame([extract_one(u)])
        outs_b[LGBM].predict(make_batch([u], d, cols, None, encode=False))
        lat.append((time.perf_counter() - t0) * 1000)
    p50, p95 = np.percentile(lat, [50, 95])
    print(f"       LightGBM end-to-end p50 = {p50:.2f} ms | p95 = {p95:.2f} ms")
    # All-model e2e latency
    urls_sample = reg_b.test.urls[:100]
    lat_rows = []
    print(f"       Measuring e2e latency for all {len(outs_b)} models (100 URLs each) ...")
    for mname, mout in outs_b.items():
        try:
            lp50, lp95 = e2e_latency(mout.predict, urls_sample, cols, reg_b.scaler, n=100)
        except Exception as _le:
            lp50, lp95 = float("nan"), float("nan")
        auc_val = float(res_b.loc[res_b["Model"] == mname, "ROC_AUC"].iloc[0]) if mname in res_b["Model"].values else float("nan")
        lat_rows.append({"Model": mname, "p50_ms": round(lp50, 3), "p95_ms": round(lp95, 3), "AUC": round(auc_val, 4)})
        print(f"         {mname:45s}  p50={lp50:.2f} ms  p95={lp95:.2f} ms  AUC={auc_val:.4f}")
    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(f"{out}/latency_summary.csv", index=False)
    print("\n       End-to-end latency summary (feature extraction + model inference):")
    print(lat_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"[9/11] {CV_FOLDS}-fold cross-validation (random stratified vs domain-grouped) ...")
    cv_summary, cv_all = (cross_validate(urls, feat, y, cols, groups, out) if CV_FOLDS and CV_FOLDS > 1
                          else (pd.DataFrame(), pd.DataFrame()))

    print("[10/11] Explainability (SHAP + LIME) and comparative graphs ...")
    lgbm_model, lgbm_cols = outs_b[LGBM].predict.model, outs_b[LGBM].predict.cols
    sv, sample = explain_global(lgbm_model, lgbm_cols, reg_b, plots, out)
    if RUN_LIME:
        run_lime(lgbm_model, lgbm_cols, reg_b, sv, sample, plots)
    make_plots(reg_b, outs_b, plots)
    final_key = ENS if (SHIP_ENSEMBLE and ENS in outs_b) else LGBM
    make_all_graphs(res_b, res_a, slice_tbl, res_x, cv_all, fs_tbl, outs_b, reg_b, plots, final_key)

    # --- Gap 5 quality: explanation coverage ---
    import shap as _shap
    try:
        _sv_all = _shap.TreeExplainer(lgbm_model).shap_values(reg_b.test.X[lgbm_cols].sample(
            min(1000, len(reg_b.test)), random_state=SEED))
        if isinstance(_sv_all, list):
            _sv_all = _sv_all[-1]
        _sv_all = np.asarray(_sv_all)
        if _sv_all.ndim == 3:
            _sv_all = _sv_all[..., -1]
        # For each test URL, check if any FEATURE_GROUP has |SHAP| > 0.15
        def _has_strong_reason(sv_row):
            for g, fs in FEATURE_GROUPS.items():
                idxs = [lgbm_cols.index(f) for f in fs if f in lgbm_cols]
                if idxs and abs(sv_row[idxs].sum()) > 0.15:
                    return True
            return False
        coverage = float(np.mean([_has_strong_reason(_sv_all[i]) for i in range(len(_sv_all))]))
        print(f"\n  Gap 5 quality — Explanation coverage: {coverage:.3f} "
              f"(fraction of test URLs with >= 1 non-generic SHAP reason, |SHAP|>0.15)")
    except Exception as _ge:
        print(f"  Gap 5 quality: could not compute explanation coverage ({_ge})")

    print("[11/11] Saving tables and artifacts ...")
    res_b.to_csv(f"{out}/results_domain_disjoint.csv", index=False)
    res_a.to_csv(f"{out}/results_random_split.csv", index=False)
    gap_tbl.to_csv(f"{out}/gap1_random_vs_domain.csv", index=False)
    res_b[["Model", "Accuracy", "Precision", "TPR", "TNR", "F1", "ROC_AUC", "PR_AUC", "Balanced_Acc", "MCC",
           "Brier_Iso", "ECE_Iso"]].to_csv(f"{out}/key_metrics_domain_disjoint.csv", index=False)
    if len(slice_tbl):
        slice_tbl.to_csv(f"{out}/gap3_shared_host_slice.csv", index=False)
    if len(res_x):
        res_x.to_csv(f"{out}/results_cross_dataset.csv", index=False)

    final_key = ENS if (SHIP_ENSEMBLE and ENS in outs_b) else LGBM
    joblib.dump({"lgbm": lgbm_model, "iso_lgbm": outs_b[LGBM].iso, "iso_cnn": outs_b[CNN].iso,
                 "iso_final": outs_b[final_key].iso, "use_ens": final_key == ENS, "cols": cols,
                 "scaler": reg_b.scaler, "latency_p50_ms": float(p50), "latency_p95_ms": float(p95)},
                f"{out}/trust_artifacts.joblib")
    torch.save(outs_b[CNN].predict.model.state_dict(), f"{out}/char_cnn.pt")
    with open(f"{out}/feature_groups.json", "w") as fh:
        json.dump(FEATURE_GROUPS, fh, indent=2)

    # ---------------- print EVERY metric ----------------
    print_confusion_summary(res_b, "TP / TN / FP / FN / TPR / TNR - DOMAIN-DISJOINT TEST SET (unseen domains)")
    print_confusion_summary(res_a, "TP / TN / FP / FN / TPR / TNR - RANDOM SPLIT")
    print_all_metrics(res_b, "ALL METRICS - DOMAIN-DISJOINT TEST SET (unseen domains)")
    print_all_metrics(res_a, "ALL METRICS - RANDOM SPLIT (inflated; for comparison)")
    print("\n=== GAP 1: Unseen-domain generalization ===")
    # Build 3-column gap table
    if len(res_x):
        gap3 = res_x[["Model", "F1"]].rename(columns={"F1": "F1_crossdataset"})
        gap_tbl = gap_tbl.merge(gap3, on="Model", how="left")
    print(gap_tbl.to_string(index=False, float_format=ff))
    print()
    for _, gr in gap_tbl.iterrows():
        drop = gr["F1_drop"]
        if not pd.isna(drop) and drop > 0.05:
            print(f"  WARNING: >5% F1 drop detected for {gr['Model']} (drop={drop:.4f}) — domain memorisation likely")
    # --- Gap 2: best calibrated model summary ---
    if "ECE_Iso" in res_b.columns:
        b2 = res_b.dropna(subset=["ECE_Iso"]).copy()
        if len(b2):
            best_cal_idx = b2["ECE_Iso"].idxmin()
            bcm = b2.loc[best_cal_idx]
            mce_val = bcm.get("MCE_Iso", float("nan"))
            print(f"\nBest calibrated model (lowest ECE_Iso): {bcm['Model']} "
                  f"(ECE={bcm['ECE_Iso']:.4f}, MCE={mce_val:.4f}, Brier={bcm['Brier_Iso']:.4f})")

    if len(slice_tbl):
        print_confusion_summary(slice_tbl, "TP / TN / FP / FN / TPR / TNR - SHARED-HOSTING SLICE")
        print_all_metrics(slice_tbl, "ALL METRICS - SHARED-HOSTING SLICE")
    if len(res_x):
        print_confusion_summary(res_x, "TP / TN / FP / FN / TPR / TNR - CROSS-DATASET")
        print_all_metrics(res_x, "ALL METRICS - CROSS-DATASET")
    if len(cv_summary):
        print("\n" + "=" * 100 + f"\n{CV_FOLDS}-FOLD CROSS-VALIDATION (mean +/- std over folds)\n" + "=" * 100)
        for scheme, sub in cv_summary.groupby(level=0):
            print(f"\n--- {scheme} ---")
            print(sub.droplevel(0).T.to_string())
    print_best_report(reg_b, outs_b, res_b[res_b["Model"].isin(outs_b)])

    print("\n=== Demo: explained trust score ===")
    art = load_artifacts(out)
    for i in [int(np.where(reg_b.test.y == 1)[0][0]), int(np.where(reg_b.test.y == 0)[0][0])]:
        show_score(reg_b.test.urls[i], art)
    print(f"\nAll tables and graphs (PCA, t-SNE, ROC/PR, reliability, comparative metrics, TPR/TNR, confusion matrices, "
          f"SHAP, LIME) saved in: {out}/ and {plots}/")


# ==============================================================================
# 8. SCORE COMMAND
# ==============================================================================
def load_artifacts(folder: str) -> dict:
    art = joblib.load(f"{folder}/trust_artifacts.joblib")
    cnn = CharCNN1D().to(DEVICE)
    cnn.load_state_dict(torch.load(f"{folder}/char_cnn.pt", map_location=DEVICE))
    art["cnn"] = cnn
    return art


def show_score(url: str, art: dict):
    lgbm_fn = lambda b: art["lgbm"].predict_proba(b.X[art["cols"]])[:, 1]
    cnn_fn = lambda b: predict_torch(art["cnn"], b)
    r = score_with_models(url, art["cols"], lgbm_fn, cnn_fn, art["iso_final"], art["iso_lgbm"], art["iso_cnn"],
                          art["scaler"], art["use_ens"])
    reasons = explain(art["lgbm"], r["X_row"], art["cols"])
    print("-" * 78)
    print("Explanation scope: LightGBM component (the CNN component is not covered by these SHAP reasons).")
    print(f"URL        : {url}")
    print(f"Trust score: {r['final_trust']}/100 ({r['band']})   [model-only: {r['calibrated_trust']}]")
    print(f"Confidence : {r['confidence']} (model disagreement {r['model_disagreement']})")
    for n in r["rule_notes"]:
        print(f"Rule       : {n}")
    for s in reasons:
        print(f"  * {s}")
    print("This is a risk estimate based on the address only, not a guarantee of safety.")
    card = generate_report_card(url, r["features"], reasons, r["final_trust"], r["band"])
    print(card)


def cmd_score(args):
    show_score(args.url, load_artifacts(args.artifacts))


# ==============================================================================
# 9. RUN  (press Run in Spyder -> trains. To score a URL, see SCORE_URL below.)
# ==============================================================================
DATA_FILE = "PhiUSIIL_Phishing_URL_Dataset.csv"   # put the full path here if the CSV is elsewhere
OUT_FOLDER = "artifacts"
EXTERNAL_FILE = None                              # optional: CSV with columns url,label (1 = phishing)
SCORE_URL = None                                  # e.g. "https://paypal-login.vercel.app/verify" -> scores instead of training
QUICK_TEST_ROWS = None                            # e.g. 20000 for a fast test run; None = full dataset


def main():
    if SCORE_URL:
        show_score(SCORE_URL, load_artifacts(OUT_FOLDER))
    else:
        cmd_train(argparse.Namespace(data=DATA_FILE, external=EXTERNAL_FILE, out=OUT_FOLDER))


if __name__ == "__main__":
    main()