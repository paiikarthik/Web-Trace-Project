# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:42:23 2026

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

Usage
-----
  python trust_pipeline.py train --data PhiUSIIL_Phishing_URL_Dataset.csv --out artifacts
  python trust_pipeline.py train --data PhiUSIIL_Phishing_URL_Dataset.csv --external my_test.csv
  python trust_pipeline.py score --url "https://paypal-login.vercel.app/verify" --artifacts artifacts

Dependencies
------------
  pip install pandas numpy scikit-learn xgboost lightgbm catboost shap torch tldextract joblib
  pip install transformers   # only if you enable DistilBERT

External test CSV (optional, for the cross-dataset regime): columns `url,label` with label 1 = phishing, 0 = legit.
PhiUSIIL has no timestamps, so a time split needs your own dated feed (PhishTank/OpenPhish + Tranco).
"""
from __future__ import annotations

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
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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
    }


def extract_features(urls) -> pd.DataFrame:
    return pd.DataFrame([extract_one(u) for u in urls])


# Ablation: features that encode the hosting-platform signal
HOSTING_FEATURES = ["is_shared_host", "shared_x_risky"]

# Gap 5: every numeric feature belongs to exactly one human-readable group.
FEATURE_GROUPS = {
    "random_name": ["host_entropy", "sub_entropy", "label_entropy", "label_digit_ratio", "digit_ratio",
                    "label_len", "label_hyphens", "letter_ratio"],
    "free_hosting": ["is_shared_host", "shared_x_risky"],
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


def make_batch(urls, feat_df: pd.DataFrame, cols: List[str], scaler: Optional[StandardScaler], y=None) -> Batch:
    X = feat_df[cols + ["TLD_cat"]].reset_index(drop=True)
    tab = scaler.transform(X[cols]).astype(np.float32) if scaler is not None else np.zeros((len(X), len(cols)), np.float32)
    seq, lens = encode_urls(list(urls))
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


def build_regime(name: str, urls, feat_df, y, cols, split) -> Regime:
    fit, val, cal, test = split
    scaler = StandardScaler().fit(feat_df.iloc[fit][cols])
    full = make_batch(urls, feat_df, cols, scaler, y)
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
    for ep in range(epochs):
        model.train()
        for s, l, t, y in loader:
            s, l, t, y = s.to(DEVICE), l.to(DEVICE), t.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(s, l, t), y).backward()
            opt.step()
        vloss = log_loss(reg.val.y, np.clip(predict_torch(model, reg.val), 1e-6, 1 - 1e-6))
        print(f"        epoch {ep + 1}/{epochs}  val loss {vloss:.4f}", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)


def torch_trainer(build: Callable[[Regime], nn.Module]):
    def trainer(reg: Regime):
        torch.manual_seed(SEED)
        model = build(reg)
        fit_torch(model, reg)
        return _wrap(lambda b: predict_torch(model, b), model)
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
}
LGBM, CNN, ENS = "4. LightGBM", "7. Char-CNN", "11. Ensemble LGBM+CNN"


# ==============================================================================
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


def _safe(fn, *a):
    try:
        return float(fn(*a))
    except ValueError:
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
        "FPR@0.5": fpr, "TPR@0.5": tpr, "FPR@TPR95": fpr95, "TPR_at_thr95": tpr95,
        "Latency_ms_single": latency_ms,
    }
    for prev in PHISH_PREVALENCES:
        row[f"Precision@{int(prev * 100)}%phish"] = precision_at_prevalence(tpr, fpr, prev)
    return row


def single_latency(pred, batch: Batch, n=200) -> float:
    n = min(n, len(batch))
    singles = [batch.take([i]) for i in range(n)]
    pred(singles[0])                                         # warm-up
    t0 = time.perf_counter()
    for s in singles:
        pred(s)
    return (time.perf_counter() - t0) / n * 1000.0


def run_models(reg: Regime, names: List[str]) -> Dict[str, ModelOut]:
    outs = {}
    for name in names:
        print(f"   -> [{reg.name}] training {name} ...", flush=True)
        pred = TRAINERS[name](reg)
        p_cal, p_test = pred(reg.cal), pred(reg.test)
        iso, platt, thr = fit_calibrators(p_cal, reg.cal.y)
        lat = single_latency(pred, reg.test, n=40 if "DistilBERT" in name else 200)
        outs[name] = ModelOut(pred, p_cal, p_test, iso, platt, thr, lat)
    return outs


def add_ensemble(reg: Regime, outs: Dict[str, ModelOut]):
    if LGBM in outs and CNN in outs:
        a, c = outs[LGBM], outs[CNN]
        pred = _wrap(lambda b: 0.5 * (a.predict(b) + c.predict(b)), None)
        p_cal, p_test = 0.5 * (a.p_cal + c.p_cal), 0.5 * (a.p_test + c.p_test)
        iso, platt, thr = fit_calibrators(p_cal, reg.cal.y)
        outs[ENS] = ModelOut(pred, p_cal, p_test, iso, platt, thr, a.latency_ms + c.latency_ms)


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
# 7. TRAIN COMMAND
# ==============================================================================
def load_dataset(path: str):
    df = pd.read_csv(path)
    df["_canon"] = df["URL"].map(lambda u: canonicalize(u).lower())
    df = df.drop_duplicates(subset="_canon").reset_index(drop=True)
    if QUICK_TEST_ROWS:
        df = df.sample(min(QUICK_TEST_ROWS, len(df)), random_state=SEED).reset_index(drop=True)
    y = (df["label"] == 0).astype(int).values            # PhiUSIIL: 1 = legit, 0 = phishing  ->  1 = phishing
    return df["URL"].tolist(), y


def cmd_train(args):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(args.out, exist_ok=True)

    print(f"[1/8] Loading {args.data} (device: {DEVICE})")
    urls, y = load_dataset(args.data)
    print(f"      {len(urls):,} unique URLs | phishing share {y.mean():.3f}")

    print("[2/8] Extracting features (PSL-aware, tenant-level) ...")
    feat = extract_features(urls)
    cols = numeric_columns(feat)
    check_groups(cols)
    groups = feat["group_domain"].values
    print(f"      {len(cols)} numeric features | {len(set(groups)):,} groups | "
          f"shared-host rows: {int(feat['is_shared_host'].sum()):,}")

    all_names = [n for n in TRAINERS if (RUN_DISTILBERT or "DistilBERT" not in n) and n not in SKIP_MODELS]

    print("[3/8] Regime B: domain-disjoint split (primary) ...")
    reg_b = build_regime("Domain-disjoint", urls, feat, y, cols, grouped_split(groups, SEED))
    print(f"      fit {len(reg_b.fit):,} | val {len(reg_b.val):,} | cal {len(reg_b.cal):,} | test {len(reg_b.test):,}")
    outs_b = run_models(reg_b, all_names)
    add_ensemble(reg_b, outs_b)
    res_b = evaluate_regime(reg_b, outs_b)

    print("[4/8] Regime A: random split (what most papers report) ...")
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

    print("[5/8] Gap 3: shared-hosting slice + hosting-feature ablation ...")
    mask = reg_b.test.X["is_shared_host"].values == 1
    ym = reg_b.test.y[mask]
    print(f"      slice size {int(mask.sum()):,} (phish {int(ym.sum()):,}, legit {int((1 - ym).sum()):,})")
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
        print("      Slice too small / single-class: collect more *.vercel.app, *.netlify.app, ... tenant URLs.")

    print("[6/8] Cross-dataset regime ...")
    res_x = pd.DataFrame()
    if args.external and os.path.exists(args.external):
        ext = pd.read_csv(args.external)
        ext_urls, ext_y = ext["url"].astype(str).tolist(), ext["label"].astype(int).values
        ext_batch = make_batch(ext_urls, extract_features(ext_urls), cols, reg_b.scaler, ext_y)
        rows = []
        for name, o in outs_b.items():
            rows.append(metrics_row("Cross-dataset", name, ext_y, o.predict(ext_batch), o.iso, o.platt, o.thr_tpr95))
        res_x = pd.DataFrame(rows)
    else:
        print("      (skipped: pass --external file.csv with columns url,label; 1 = phishing)")

    print("[7/8] Gap 4: end-to-end latency (feature extraction + LightGBM) ...")
    sample = reg_b.test.urls[:500]
    lat = []
    for u in sample:
        t0 = time.perf_counter()
        d = pd.DataFrame([extract_one(u)])
        outs_b[LGBM].predict(make_batch([u], d, cols, None))
        lat.append((time.perf_counter() - t0) * 1000)
    p50, p95 = np.percentile(lat, [50, 95])
    print(f"      end-to-end p50 = {p50:.2f} ms | p95 = {p95:.2f} ms")

    print("[8/8] Saving tables and artifacts ...")
    res_b.to_csv(f"{args.out}/results_domain_disjoint.csv", index=False)
    res_a.to_csv(f"{args.out}/results_random_split.csv", index=False)
    gap_tbl.to_csv(f"{args.out}/gap1_random_vs_domain.csv", index=False)
    if len(slice_tbl):
        slice_tbl.to_csv(f"{args.out}/gap3_shared_host_slice.csv", index=False)
    if len(res_x):
        res_x.to_csv(f"{args.out}/results_cross_dataset.csv", index=False)

    final_key = ENS if (SHIP_ENSEMBLE and ENS in outs_b) else LGBM
    joblib.dump({"lgbm": outs_b[LGBM].predict.model, "iso_lgbm": outs_b[LGBM].iso, "iso_cnn": outs_b[CNN].iso,
                 "iso_final": outs_b[final_key].iso, "use_ens": final_key == ENS, "cols": cols,
                 "scaler": reg_b.scaler, "latency_p50_ms": float(p50), "latency_p95_ms": float(p95)},
                f"{args.out}/trust_artifacts.joblib")
    torch.save(outs_b[CNN].predict.model.state_dict(), f"{args.out}/char_cnn.pt")
    with open(f"{args.out}/feature_groups.json", "w") as fh:
        json.dump(FEATURE_GROUPS, fh, indent=2)

    show = ["Model", "F1", "PR_AUC", "Brier_Raw", "Brier_Iso", "ECE_Raw", "ECE_Iso", "FPR@0.5", "FPR@TPR95",
            "Precision@5%phish", "Precision@1%phish", "Latency_ms_single"]
    pd.set_option("display.width", 250)
    print("\n=== Domain-disjoint results ===")
    print(res_b[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== Gap 1: random vs domain-disjoint ===")
    print(gap_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(slice_tbl):
        print("\n=== Gap 3: shared-hosting slice ===")
        print(slice_tbl[["Model", "N", "Phish_share", "F1", "PR_AUC", "FPR@0.5", "TPR@0.5"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(res_x):
        print("\n=== Cross-dataset ===")
        print(res_x[["Model", "N", "F1", "PR_AUC", "FPR@0.5", "ECE_Iso"]].to_string(index=False,
                                                                                    float_format=lambda v: f"{v:.4f}"))

    print("\n=== Demo: explained trust score ===")
    art = load_artifacts(args.out)
    for i in [int(np.where(reg_b.test.y == 1)[0][0]), int(np.where(reg_b.test.y == 0)[0][0])]:
        show_score(reg_b.test.urls[i], art)


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
    print(f"URL        : {url}")
    print(f"Trust score: {r['final_trust']}/100 ({r['band']})   [model-only: {r['calibrated_trust']}]")
    print(f"Confidence : {r['confidence']} (model disagreement {r['model_disagreement']})")
    for n in r["rule_notes"]:
        print(f"Rule       : {n}")
    for s in reasons:
        print(f"  * {s}")
    print("This is a risk estimate based on the address only, not a guarantee of safety.")


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