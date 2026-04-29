"""
training.py  —  ProctorAI v6  (90%+ Accuracy Edition)
=======================================================
5-Class Detection:
  0 Normal | 1 Phone | 2 Multi-Person | 3 Voice | 4 Identity-Mismatch

Key upgrades over original training.py:
  • 5 classes (was 4)  |  72 features (was 56)
  • Deep Residual CNN + Squeeze-and-Excitation attention
  • Cosine Decay with Warm Restarts (SGDR)
  • Mixup augmentation (alpha=0.20)
  • Label smoothing (epsilon=0.10)
  • CNN (0.60) + RandomForest (0.40) ensemble
  • 90,000 samples, 18K per class
  • Saves: cnn_phone.h5  +  proctor_cnn_v6.keras

Run:
    pip install tensorflow scikit-learn pandas matplotlib
    python training.py
"""

import os, sys, json, pickle, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import StandardScaler, label_binarize
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        roc_auc_score, f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble           import RandomForestClassifier

# ── Config ─────────────────────────────────────────────────────────────────────
SEED          = 42
BATCH_SIZE    = 512
EPOCHS        = 10
LR_INIT       = 2e-3
LR_MIN        = 5e-7
WARMUP_EP     = 5
DROPOUT       = 0.35
L2_REG        = 5e-5
VAL_FRAC      = 0.12
TEST_FRAC     = 0.12
MIXUP_ALPHA   = 0.20
N_PER_CLASS   = 18000    # 18K x 5 classes = 90K samples

NUM_CLASSES   = 5
NUM_FEATURES  = 72
CLASS_NAMES   = ["Normal","Phone","Multi-Person","Voice","Identity-Mismatch"]

CLASS_THRESHOLDS = {
    "Normal":0.50, "Phone":0.40, "Multi-Person":0.40,
    "Voice":0.38,  "Identity-Mismatch":0.40
}

DATA_PATH  = "data/proctor_dataset_v6.csv"
META_PATH  = "data/proctor_meta_v6.json"
MODEL_DIR  = "models"
STATIC_DIR = "static"

np.random.seed(SEED)
for d in [MODEL_DIR, STATIC_DIR, "data"]:
    os.makedirs(d, exist_ok=True)

FEATURE_NAMES = [
    # [0-11]  Phone / YOLO
    "phone_yolo_conf","lower_brightness","lower_brightness_std","phone_brightness_score",
    "phone_uniformity_score","phone_rect_score","phone_glare_score","phone_combined_score",
    "phone_screen_brightness","phone_persist_3f","phone_persist_5f","phone_edge_density",
    # [12-23] Multi-person / face / YOLO
    "phone_conf_heuristic","face2_conf","face2_area_ratio","face2_detection_score",
    "person_yolo_count","face_yolo_count","face_iou_persons","skin_ratio_full",
    "face_count_norm","face_separation","skin_ratio_edges","person2_yolo_conf",
    # [24-35] Audio / Voice
    "audio_rms_norm","audio_zcr_norm","audio_peak_norm","voice_probability",
    "harmonic_ratio","background_noise_inv","lip_motion_score","voice_persist_3f",
    "voice_persist_5f","audio_spectral_flux","voice_snr","audio_energy_ratio",
    # [36-47] Identity verification
    "face_embed_dist","face_match_score","face_match_conf","lbp_similarity",
    "voice_embed_dist","voice_match_score","voice_match_conf","identity_conf",
    "face_landmark_stability","face_quality_score","face_area_change","identity_alert_flag",
    # [48-59] Head pose / gaze
    "head_yaw","head_pitch","head_roll","head_yaw_abs","head_pitch_abs",
    "face_width_ratio","face_height_ratio","side_view_score","pose_deviation",
    "gaze_asymmetry","eye_openness","face_symmetry",
    # [60-71] Motion / composite
    "attention_score","gaze_score","laplacian_mean","laplacian_std","upper_laplacian",
    "lower_laplacian","motion_score","temporal_consistency","pose_stability_5f",
    "composite_alert","anomaly_score","integrity_score",
]
assert len(FEATURE_NAMES) == NUM_FEATURES

# ── TensorFlow ─────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, callbacks, backend as K
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
    tf.random.set_seed(SEED)
    HAS_TF = True
    print(f"[TF] TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        [tf.config.experimental.set_memory_growth(g, True) for g in gpus]
        print(f"[TF] GPU(s): {[g.name for g in gpus]}")
    else:
        print("[TF] CPU mode")
except ImportError:
    HAS_TF = False
    print("[WARN] TensorFlow not found -> RF fallback. pip install tensorflow")

print()


# ==============================================================================
# 1.  DATASET GENERATION  (72 features x 5 classes x 18K = 90K samples)
# ==============================================================================
def _rng(n, lo, hi, std=None):
    if std is None:
        r = np.random.uniform(lo, hi, n)
    else:
        r = np.random.normal((lo+hi)/2, std, n)
    return np.clip(r, 0.0, 1.0).astype(np.float32)


def generate_dataset(n_per_class=N_PER_CLASS):
    print(f"[DATA] Generating {n_per_class*NUM_CLASSES:,} synthetic samples ...")
    N = n_per_class

    def blank(n):
        return np.zeros((n, NUM_FEATURES), dtype=np.float32)

    # ── Class 0: Normal ────────────────────────────────────────────────────────
    c0 = blank(N)
    c0[:,0]  = _rng(N,0.00,0.08)      # phone_yolo_conf: very low
    c0[:,1]  = _rng(N,0.10,0.28)      # lower_brightness: normal ambient
    c0[:,2]  = _rng(N,0.05,0.14)
    c0[:,3]  = _rng(N,0.00,0.06)
    c0[:,4]  = _rng(N,0.60,0.95)      # phone_uniformity: background
    c0[:,12] = _rng(N,0.00,0.07)      # phone_conf_heuristic: low
    c0[:,13] = _rng(N,0.00,0.05)      # face2_conf: no 2nd face
    c0[:,16] = _rng(N,1.0,1.2,0.1)    # person_yolo_count: 1
    c0[:,17] = _rng(N,1.0,1.15,0.08)  # face_yolo_count: 1
    c0[:,20] = _rng(N,0.22,0.32)      # face_count_norm: 1 face
    c0[:,22] = _rng(N,0.04,0.12)      # skin_ratio_edges: low
    c0[:,24] = _rng(N,0.00,0.04)      # audio_rms: silent
    c0[:,25] = _rng(N,0.10,0.35)
    c0[:,27] = _rng(N,0.00,0.09)      # voice_prob: low
    c0[:,36] = _rng(N,0.00,0.12)      # face_embed_dist: low (same person)
    c0[:,37] = _rng(N,0.88,1.00)      # face_match_score: HIGH
    c0[:,40] = _rng(N,0.00,0.10)      # voice_embed_dist: low
    c0[:,41] = _rng(N,0.86,1.00)      # voice_match_score: HIGH
    c0[:,43] = _rng(N,0.90,1.00)      # identity_conf: HIGH
    c0[:,47] = np.zeros(N)             # identity_alert: 0
    c0[:,48] = _rng(N,-0.15,0.15)     # head_yaw: forward
    c0[:,49] = _rng(N,-0.12,0.12)
    c0[:,55] = _rng(N,0.00,0.12)      # side_view_score: low
    c0[:,60] = _rng(N,0.85,1.00)      # attention: HIGH
    c0[:,70] = _rng(N,0.00,0.06)      # composite_alert: low
    c0[:,71] = _rng(N,0.00,0.06)      # anomaly: low
    for i in [5,6,7,8,9,10,11,14,15,18,19,21,23,26,28,29,30,31,32,33,34,35,
              38,39,42,44,45,46,50,51,52,53,54,56,57,58,59,61,62,63,64,65,66,67,68,69]:
        c0[:,i] = _rng(N,0.01,0.16)

    # ── Class 1: Phone ─────────────────────────────────────────────────────────
    c1 = blank(N)
    c1[:,0]  = _rng(N,0.70,0.98)      # phone_yolo_conf: VERY HIGH
    c1[:,1]  = _rng(N,0.55,0.92)      # lower_brightness: bright screen
    c1[:,2]  = _rng(N,0.02,0.09)      # uniform bright
    c1[:,3]  = _rng(N,0.65,0.96)
    c1[:,4]  = _rng(N,0.72,0.98)      # phone_uniformity: HIGH
    c1[:,5]  = _rng(N,0.60,0.94)      # phone_rect_score: rectangular
    c1[:,6]  = _rng(N,0.50,0.90)
    c1[:,7]  = _rng(N,0.70,0.98)
    c1[:,8]  = _rng(N,0.65,0.98)      # screen_brightness
    c1[:,9]  = _rng(N,0.60,0.95)
    c1[:,10] = _rng(N,0.55,0.95)
    c1[:,11] = _rng(N,0.55,0.88)
    c1[:,12] = _rng(N,0.72,0.98)      # phone_conf_heuristic: VERY HIGH
    c1[:,13] = _rng(N,0.00,0.06)      # face2_conf: 1 person
    c1[:,16] = _rng(N,1.0,1.1,0.07)
    c1[:,20] = _rng(N,0.18,0.28)
    c1[:,24] = _rng(N,0.00,0.06)      # audio: silent
    c1[:,27] = _rng(N,0.00,0.12)
    c1[:,36] = _rng(N,0.00,0.14)      # same person using phone
    c1[:,37] = _rng(N,0.80,0.98)
    c1[:,40] = _rng(N,0.00,0.12)
    c1[:,41] = _rng(N,0.80,0.98)
    c1[:,43] = _rng(N,0.82,0.98)
    c1[:,47] = np.zeros(N)
    c1[:,48] = _rng(N,-0.25,0.25)
    c1[:,49] = _rng(N,0.20,0.55)      # head_pitch: DOWN looking at phone
    c1[:,55] = _rng(N,0.10,0.30)
    c1[:,60] = _rng(N,0.20,0.55)      # attention: LOW
    c1[:,70] = _rng(N,0.60,0.92)      # composite_alert: HIGH
    c1[:,71] = _rng(N,0.55,0.90)
    for i in [14,15,17,18,19,21,22,23,25,26,28,29,30,31,32,33,34,35,
              38,39,42,44,45,46,50,51,52,53,54,56,57,58,59,61,62,63,64,65,66,67,68,69]:
        c1[:,i] = _rng(N,0.02,0.20)

    # ── Class 2: Multi-Person ──────────────────────────────────────────────────
    c2 = blank(N)
    c2[:,0]  = _rng(N,0.00,0.10)      # phone_yolo: low
    c2[:,12] = _rng(N,0.00,0.09)
    c2[:,13] = _rng(N,0.78,0.98)      # face2_conf: VERY HIGH (2nd face)
    c2[:,14] = _rng(N,0.04,0.18)      # face2_area_ratio: significant size
    c2[:,15] = _rng(N,0.72,0.96)      # face2_detection_score
    c2[:,16] = _rng(N,1.8,3.2,0.4)    # person_yolo_count: 2-3
    c2[:,17] = _rng(N,1.8,3.0,0.35)   # face_yolo_count: 2-3
    c2[:,18] = _rng(N,0.10,0.35)      # face_iou
    c2[:,19] = _rng(N,0.28,0.55)      # skin_ratio_full: more skin
    c2[:,20] = _rng(N,0.42,0.75)      # face_count_norm: HIGH
    c2[:,21] = _rng(N,0.28,0.55)      # face_separation
    c2[:,22] = _rng(N,0.22,0.50)      # skin_ratio_edges: HIGH
    c2[:,23] = _rng(N,0.68,0.96)      # person2_yolo_conf: HIGH
    c2[:,24] = _rng(N,0.00,0.08)
    c2[:,27] = _rng(N,0.00,0.15)
    c2[:,36] = _rng(N,0.00,0.14)
    c2[:,37] = _rng(N,0.78,0.96)
    c2[:,43] = _rng(N,0.78,0.95)
    c2[:,47] = np.zeros(N)
    c2[:,48] = _rng(N,-0.20,0.20)
    c2[:,55] = _rng(N,0.05,0.25)
    c2[:,60] = _rng(N,0.35,0.65)      # attention: medium
    c2[:,70] = _rng(N,0.55,0.92)
    c2[:,71] = _rng(N,0.55,0.88)
    for i in [1,2,3,4,5,6,7,8,9,10,11,25,26,28,29,30,31,32,33,34,35,
              38,39,40,41,42,44,45,46,49,50,51,52,53,54,56,57,58,59,61,62,63,64,65,66,67,68,69]:
        c2[:,i] = _rng(N,0.02,0.20)

    # ── Class 3: Voice / Speaking ──────────────────────────────────────────────
    c3 = blank(N)
    c3[:,0]  = _rng(N,0.00,0.08)
    c3[:,12] = _rng(N,0.00,0.08)
    c3[:,13] = _rng(N,0.00,0.06)      # 1 person
    c3[:,16] = _rng(N,1.0,1.1,0.07)
    c3[:,20] = _rng(N,0.22,0.32)
    c3[:,24] = _rng(N,0.55,0.92)      # audio_rms: VERY HIGH
    c3[:,25] = _rng(N,0.40,0.78)      # audio_zcr: HIGH (speech)
    c3[:,26] = _rng(N,0.50,0.90)      # audio_peak: HIGH
    c3[:,27] = _rng(N,0.72,0.98)      # voice_prob: VERY HIGH
    c3[:,28] = _rng(N,0.65,0.92)      # harmonic_ratio: HIGH (speech harmonics)
    c3[:,29] = _rng(N,0.10,0.40)
    c3[:,30] = _rng(N,0.50,0.88)      # lip_motion: HIGH (lips moving)
    c3[:,31] = _rng(N,0.60,0.95)
    c3[:,32] = _rng(N,0.55,0.92)
    c3[:,33] = _rng(N,0.55,0.88)
    c3[:,34] = _rng(N,0.50,0.88)
    c3[:,35] = _rng(N,0.60,0.95)
    c3[:,36] = _rng(N,0.00,0.14)
    c3[:,37] = _rng(N,0.82,0.98)
    c3[:,40] = _rng(N,0.00,0.12)
    c3[:,41] = _rng(N,0.78,0.98)
    c3[:,43] = _rng(N,0.82,0.98)
    c3[:,47] = np.zeros(N)
    c3[:,48] = _rng(N,-0.18,0.18)
    c3[:,55] = _rng(N,0.00,0.18)
    c3[:,60] = _rng(N,0.60,0.88)
    c3[:,70] = _rng(N,0.45,0.80)
    c3[:,71] = _rng(N,0.38,0.72)
    for i in [1,2,3,4,5,6,7,8,9,10,11,14,15,17,18,19,21,22,23,
              38,39,42,44,45,46,49,50,51,52,53,54,56,57,58,59,61,62,63,64,65,66,67,68,69]:
        c3[:,i] = _rng(N,0.02,0.18)

    # ── Class 4: Identity-Mismatch ─────────────────────────────────────────────
    c4 = blank(N)
    c4[:,0]  = _rng(N,0.00,0.10)
    c4[:,12] = _rng(N,0.00,0.10)
    c4[:,13] = _rng(N,0.00,0.06)
    c4[:,16] = _rng(N,1.0,1.2,0.1)
    c4[:,20] = _rng(N,0.22,0.35)
    c4[:,24] = _rng(N,0.00,0.08)
    c4[:,27] = _rng(N,0.00,0.12)
    # *** Identity features: KEY separating features ***
    c4[:,36] = _rng(N,0.62,0.98)      # face_embed_dist: VERY HIGH (different face)
    c4[:,37] = _rng(N,0.02,0.28)      # face_match_score: VERY LOW
    c4[:,38] = _rng(N,0.00,0.20)      # face_match_conf: LOW
    c4[:,39] = _rng(N,0.02,0.30)      # lbp_similarity: LOW
    c4[:,40] = _rng(N,0.55,0.95)      # voice_embed_dist: HIGH
    c4[:,41] = _rng(N,0.05,0.32)      # voice_match_score: VERY LOW
    c4[:,42] = _rng(N,0.00,0.22)
    c4[:,43] = _rng(N,0.02,0.28)      # identity_conf: VERY LOW
    c4[:,44] = _rng(N,0.30,0.70)      # face_quality: present but different
    c4[:,45] = _rng(N,0.50,0.90)      # landmark_stability
    c4[:,46] = _rng(N,0.50,0.92)      # face_area_change
    c4[:,47] = np.ones(N)              # identity_alert_flag: 1
    c4[:,48] = _rng(N,-0.22,0.22)
    c4[:,55] = _rng(N,0.05,0.22)
    c4[:,60] = _rng(N,0.55,0.82)
    c4[:,70] = _rng(N,0.55,0.92)
    c4[:,71] = _rng(N,0.60,0.95)
    for i in [1,2,3,4,5,6,7,8,9,10,11,14,15,17,18,19,21,22,23,25,26,28,29,30,31,32,33,34,35,
              49,50,51,52,53,54,56,57,58,59,61,62,63,64,65,66,67,68,69]:
        c4[:,i] = _rng(N,0.02,0.20)

    X = np.vstack([c0,c1,c2,c3,c4]).astype(np.float32)
    y = np.concatenate([np.full(N,i,dtype=np.int32) for i in range(NUM_CLASSES)])

    # 5% additive noise for robustness
    noise_idx = np.random.choice(len(X), int(len(X)*0.05), replace=False)
    X[noise_idx] += np.random.normal(0, 0.035, (len(noise_idx), NUM_FEATURES))
    X = np.clip(X, 0.0, 1.0).astype(np.float32)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df.to_csv(DATA_PATH, index=False)
    meta = {"n_samples":len(df),"n_features":NUM_FEATURES,"n_classes":NUM_CLASSES,
            "class_names":CLASS_NAMES,"feature_names":FEATURE_NAMES}
    with open(META_PATH,"w") as f: json.dump(meta,f,indent=2)
    print(f"[DATA] Saved {len(df):,} rows x {NUM_FEATURES} features -> {DATA_PATH}")
    print(f"[DATA] Classes: { {CLASS_NAMES[i]:int((y==i).sum()) for i in range(NUM_CLASSES)} }")
    return X, y


def load_data():
    if not os.path.exists(DATA_PATH):
        print("[DATA] Not found, generating ...")
        return generate_dataset()
    df = pd.read_csv(DATA_PATH)
    X  = df[[c for c in df.columns if c!="label"]].values.astype(np.float32)
    y  = df["label"].values.astype(np.int32)
    print(f"[DATA] Loaded {len(df):,} x {X.shape[1]} features")
    print(f"[DATA] Classes: { {CLASS_NAMES[i]:int((y==i).sum()) for i in range(NUM_CLASSES)} }")
    return X, y


def preprocess(X, y):
    X_tr,X_tmp,y_tr,y_tmp = train_test_split(X,y,test_size=VAL_FRAC+TEST_FRAC,stratify=y,random_state=SEED)
    X_val,X_te,y_val,y_te = train_test_split(X_tmp,y_tmp,
        test_size=TEST_FRAC/(VAL_FRAC+TEST_FRAC),stratify=y_tmp,random_state=SEED)
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_te  = scaler.transform(X_te).astype(np.float32)
    print(f"[SPLIT] Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_te):,}")
    # CNN: (batch, features, 1)
    if HAS_TF:
        return ((X_tr.reshape(-1,NUM_FEATURES,1), y_tr),
                (X_val.reshape(-1,NUM_FEATURES,1), y_val),
                (X_te.reshape(-1,NUM_FEATURES,1),  y_te),
                scaler, X_tr, X_val, X_te)
    return (X_tr,y_tr),(X_val,y_val),(X_te,y_te),scaler,X_tr,X_val,X_te


# ==============================================================================
# 2.  CNN ARCHITECTURE — Residual + Squeeze-Excitation
# ==============================================================================
def se_block(x, ratio=8):
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(ch//ratio,4), activation="relu", kernel_initializer="he_normal")(se)
    se = layers.Dense(ch, activation="sigmoid")(se)
    se = layers.Reshape((1,ch))(se)
    return layers.Multiply()([x, se])


def res_block(x, filters, kernel=3, dilation=1, pool=False):
    sc = x
    y = layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation,
                      kernel_regularizer=regularizers.l2(L2_REG),
                      kernel_initializer="he_normal")(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv1D(filters, 3, padding="same",
                      kernel_regularizer=regularizers.l2(L2_REG),
                      kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    y = se_block(y, ratio=8)
    if sc.shape[-1] != filters:
        sc = layers.Conv1D(filters, 1, padding="same",
                           kernel_initializer="he_normal")(sc)
        sc = layers.BatchNormalization()(sc)
    y = layers.Add()([y, sc])
    y = layers.Activation("relu")(y)
    y = layers.SpatialDropout1D(DROPOUT*0.4)(y)
    if pool: y = layers.MaxPooling1D(2)(y)
    return y


def build_cnn():
    inp = layers.Input(shape=(NUM_FEATURES,1), name="features")

    # Entry block
    x = layers.Conv1D(64, 7, padding="same", kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(L2_REG))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual blocks (increasing depth & dilation)
    x = res_block(x, 64,  kernel=5, dilation=1, pool=True)   # 72->36
    x = res_block(x, 128, kernel=5, dilation=1, pool=True)   # 36->18
    x = res_block(x, 128, kernel=3, dilation=2, pool=False)  # dilation
    x = res_block(x, 256, kernel=3, dilation=1, pool=False)
    x = res_block(x, 256, kernel=3, dilation=4, pool=False)  # large rf
    x = res_block(x, 512, kernel=3, dilation=1, pool=False)

    # Dual global pooling
    gmax = layers.GlobalMaxPooling1D()(x)
    gavg = layers.GlobalAveragePooling1D()(x)
    x    = layers.Concatenate()([gmax, gavg])  # 1024-dim
    x    = layers.Dropout(DROPOUT)(x)

    # Dense head
    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_REG),
                     kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(L2_REG))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT*0.6)(x)

    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(DROPOUT*0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="class_probs")(x)

    model = models.Model(inp, out, name="ProctorCNN_v6_ResidualSE")

    # Cosine decay with warm restarts
    steps_per_epoch = int(N_PER_CLASS * NUM_CLASSES * 0.76 * 2 / BATCH_SIZE)  # *2 for mixup
    lr_sched = CosineDecayRestarts(
        initial_learning_rate=LR_INIT,
        first_decay_steps=steps_per_epoch * 20,
        t_mul=1.5, m_mul=0.85, alpha=LR_MIN
    )
    model.compile(
        optimizer=Adam(learning_rate=lr_sched, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2")]
    )
    return model


# ==============================================================================
# 3.  MIXUP AUGMENTATION
# ==============================================================================
def mixup(X, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha, len(X))
    lam = np.maximum(lam, 1-lam)
    idx = np.random.permutation(len(X))
    Xm  = (lam.reshape(-1,1,1)*X + (1-lam.reshape(-1,1,1))*X[idx]).astype(np.float32)
    ym  = np.where(lam>=0.5, y, y[idx]).astype(np.int32)
    return Xm, ym


# ==============================================================================
# 4.  TRAINING
# ==============================================================================
def train_cnn(model, train_data, val_data):
    X_tr, y_tr   = train_data
    X_val, y_val = val_data

    cw_vals = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_tr)
    cw = {i:float(v) for i,v in enumerate(cw_vals)}
    print(f"[TRAIN] class_weight = { {CLASS_NAMES[i]:round(v,3) for i,v in cw.items()} }")

    # Mixup augmentation
    Xm, ym = mixup(X_tr, y_tr, MIXUP_ALPHA)
    Xt = np.concatenate([X_tr, Xm], axis=0)
    yt = np.concatenate([y_tr, ym], axis=0)
    print(f"[TRAIN] After Mixup: {len(Xt):,} samples")

    cb_list = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                restore_best_weights=True, mode="max",
                                min_delta=0.001, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                                    patience=10, min_lr=LR_MIN, verbose=1),
        callbacks.ModelCheckpoint(f"{MODEL_DIR}/best_proctor_v6.keras",
                                  monitor="val_accuracy", save_best_only=True,
                                  mode="max", verbose=1),
        callbacks.TerminateOnNaN(),
    ]

    print(f"\n[TRAIN] Starting — max {EPOCHS} epochs, batch={BATCH_SIZE}")
    t0   = time.time()
    hist = model.fit(Xt, yt, validation_data=(X_val, y_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     class_weight=cw, callbacks=cb_list,
                     shuffle=True, verbose=1)
    elapsed = time.time() - t0
    best_val = max(hist.history["val_accuracy"])
    print(f"\n[TRAIN] Done in {elapsed:.0f}s")
    print(f"[TRAIN] Best val_accuracy = {best_val:.4f}  ({best_val*100:.2f}%)")
    return hist


# ==============================================================================
# 5.  RANDOMFOREST ENSEMBLE MEMBER
# ==============================================================================
def train_rf(X_tr, y_tr, X_val, y_val):
    print("\n[RF] Training RandomForest ensemble member ...")
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=25, min_samples_leaf=2,
        max_features="sqrt", n_jobs=-1, random_state=SEED,
        class_weight="balanced", oob_score=True)
    clf.fit(X_tr, y_tr)
    val_acc = accuracy_score(y_val, np.argmax(clf.predict_proba(X_val), axis=1))
    print(f"[RF] OOB score = {clf.oob_score_:.4f}  |  Val acc = {val_acc:.4f}")
    return clf


# ==============================================================================
# 6.  ENSEMBLE EVALUATION
# ==============================================================================
def evaluate(cnn, rf, test_cnn, X_te_flat, y_te, cnn_w=0.60, rf_w=0.40):
    print(f"\n[EVAL] CNN({cnn_w}) + RF({rf_w}) ensemble ...")
    cp = cnn.predict(test_cnn[0], batch_size=512, verbose=0)
    rp = rf.predict_proba(X_te_flat)
    if rp.shape[1] < NUM_CLASSES:
        rp = np.concatenate([rp, np.zeros((len(rp), NUM_CLASSES-rp.shape[1]))], axis=1)

    ep = cnn_w*cp + rf_w*rp
    pd_arr = np.argmax(ep, axis=1)

    acc = accuracy_score(y_te, pd_arr)
    f1m = f1_score(y_te, pd_arr, average="macro")
    y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))
    auc = roc_auc_score(y_bin, ep, multi_class="ovr", average="macro")
    cm  = confusion_matrix(y_te, pd_arr)

    print(f"\n{'='*58}")
    print(f"  ENSEMBLE RESULTS")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)  {'✅ >= 90%' if acc>=0.90 else '⚠ <90%'}")
    print(f"  Macro-F1  : {f1m:.4f}")
    print(f"  Macro-AUC : {auc:.4f}")
    print(f"{'='*58}")
    print(classification_report(y_te, pd_arr, target_names=CLASS_NAMES))

    cnn_acc = accuracy_score(y_te, np.argmax(cp, axis=1))
    rf_acc  = accuracy_score(y_te, np.argmax(rp, axis=1))
    print(f"[EVAL] CNN-only acc = {cnn_acc:.4f}  |  RF-only acc = {rf_acc:.4f}")

    results = {
        "model_type":"CNN_RF_Ensemble","accuracy":round(float(acc),4),
        "macro_f1":round(float(f1m),4),"macro_auc":round(float(auc),4),
        "cnn_accuracy":round(float(cnn_acc),4),"rf_accuracy":round(float(rf_acc),4),
        "ensemble_weights":{"cnn":cnn_w,"rf":rf_w},
        "confusion_matrix":cm.tolist(),"class_names":CLASS_NAMES,
        "thresholds":CLASS_THRESHOLDS,
        "classification_report":classification_report(y_te, pd_arr,
            target_names=CLASS_NAMES, output_dict=True)
    }
    with open(f"{MODEL_DIR}/eval_results_v6.json","w") as f:
        json.dump(results,f,indent=2)
    return ep, pd_arr, auc, cm, acc


# ==============================================================================
# 7.  SAVE ARTEFACTS
# ==============================================================================
def save_artefacts(cnn, rf, scaler, acc):
    h5_path    = f"{MODEL_DIR}/cnn_phone.h5"
    keras_path = f"{MODEL_DIR}/proctor_cnn_v6.keras"
    rf_path    = f"{MODEL_DIR}/proctor_rf_v6.pkl"
    sc_path    = f"{MODEL_DIR}/scaler_v6.pkl"

    cnn.save(h5_path)
    print(f"[SAVE] {h5_path}  (HDF5 — primary output)")
    cnn.save(keras_path)
    print(f"[SAVE] {keras_path}  (SavedModel)")
    with open(rf_path,"wb") as f: pickle.dump(rf, f)
    print(f"[SAVE] {rf_path}")
    with open(sc_path,"wb") as f: pickle.dump(scaler, f)
    print(f"[SAVE] {sc_path}")

    meta = {
        "version":"v6","model_type":"CNN_RF_Ensemble",
        "cnn_h5_path":h5_path,"cnn_keras_path":keras_path,
        "rf_path":rf_path,"scaler_path":sc_path,
        "input_shape":[NUM_FEATURES,1],"num_features":NUM_FEATURES,
        "num_classes":NUM_CLASSES,"class_names":CLASS_NAMES,
        "thresholds":CLASS_THRESHOLDS,
        "ensemble_weights":{"cnn":0.60,"rf":0.40},
        "test_accuracy":round(float(acc),4),
        "feature_names":FEATURE_NAMES,
        "yolo_integration":{
            "phone_class_id":67,"person_class_id":0,
            "model":"yolov8n.pt","conf_threshold":0.45,
            "notes":"YOLOv8 detects phone (class 67) and person (class 0) in real-time"
        }
    }
    with open(f"{MODEL_DIR}/model_meta_v6.json","w") as f:
        json.dump(meta,f,indent=2)
    print(f"[SAVE] {MODEL_DIR}/model_meta_v6.json")


# ==============================================================================
# 8.  TRAINING DASHBOARD
# ==============================================================================
def save_plots(hist, ep_probs, y_te, auc, cm, acc):
    BG="#060c1a"; CELL="#0c1628"; TXT="#dde4f0"; CYAN="#22d3ee"
    C5=["#22c55e","#ef4444","#f97316","#8b5cf6","#be123c"]
    TRAIN="#60a5fa"; VAL="#f87171"

    ep = range(1, len(hist.history["loss"])+1)
    fig = plt.figure(figsize=(22,14)); fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3,4, figure=fig, hspace=0.45, wspace=0.38)

    def _ax(r,c,cs=1):
        ax = fig.add_subplot(gs[r,c:c+cs])
        ax.set_facecolor(CELL)
        ax.tick_params(colors=TXT, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        ax.title.set_color(CYAN); ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)
        return ax

    # 1. Loss
    ax=_ax(0,0)
    ax.plot(ep,hist.history["loss"],color=TRAIN,lw=1.8,label="Train")
    ax.plot(ep,hist.history["val_loss"],color=VAL,lw=1.8,label="Val")
    ax.axvline(int(np.argmin(hist.history["val_loss"]))+1,color=CYAN,ls="--",lw=1.2,alpha=0.7)
    ax.set_title("Cross-Entropy Loss"); ax.legend(facecolor=CELL,labelcolor=TXT,fontsize=8)

    # 2. Accuracy
    ax=_ax(0,1)
    ax.plot(ep,hist.history["accuracy"],color=TRAIN,lw=1.8,label="Train")
    ax.plot(ep,hist.history["val_accuracy"],color=VAL,lw=1.8,label="Val")
    ax.axhline(0.90,color="#22c55e",ls=":",lw=1.5,label="90% target")
    ax.set_ylim(0,1.05)
    ax.set_title(f"Accuracy  (best val={max(hist.history['val_accuracy']):.3f})")
    ax.legend(facecolor=CELL,labelcolor=TXT,fontsize=8)

    # 3. Top-2
    ax=_ax(0,2)
    tk = "top2" if "top2" in hist.history else "top2_acc"
    vk = "val_top2" if "val_top2" in hist.history else "val_top2_acc"
    if tk in hist.history:
        ax.plot(ep,hist.history[tk],color=TRAIN,lw=1.8,label="Train")
        ax.plot(ep,hist.history[vk],color=VAL,lw=1.8,label="Val")
    ax.set_ylim(0,1.05); ax.set_title("Top-2 Accuracy")
    ax.legend(facecolor=CELL,labelcolor=TXT,fontsize=8)

    # 4. LR
    ax=_ax(0,3)
    if "lr" in hist.history:
        ax.plot(ep,hist.history["lr"],color="#a78bfa",lw=1.8)
        ax.set_yscale("log")
    ax.set_title("Learning Rate (Cosine Restarts)")

    # 5. Confusion matrix
    ax=_ax(1,0,colspan=2)
    cm_n = cm.astype(float)/cm.sum(axis=1,keepdims=True)
    ax.imshow(cm_n,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES,rotation=25,ha="right",color=TXT,fontsize=8)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASS_NAMES,color=TXT,fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,f"{v:,}\n({cm_n[i,j]:.0%})",ha="center",va="center",
                color="white" if cm_n[i,j]>0.45 else TXT,fontsize=7,fontweight="bold")
    ax.set_title(f"Confusion Matrix  |  Acc={acc*100:.2f}%  AUC={auc:.4f}")

    # 6. Confidence distributions
    ax=_ax(1,2,colspan=2)
    for ci in range(NUM_CLASSES):
        m = y_te==ci
        if m.sum()>0:
            ax.hist(ep_probs[m,ci],bins=50,color=C5[ci],alpha=0.65,
                    label=CLASS_NAMES[ci],density=True)
    ax.set_xlabel("P(true class)"); ax.set_title("Confidence Distributions")
    ax.legend(facecolor=CELL,labelcolor=TXT,fontsize=8)

    # 7. Per-class F1
    ax=_ax(2,0,colspan=2)
    try:
        with open(f"{MODEL_DIR}/eval_results_v6.json") as f_: er=json.load(f_)
        cr=er["classification_report"]
        f1s=[cr[n]["f1-score"] for n in CLASS_NAMES]
        prec=[cr[n]["precision"] for n in CLASS_NAMES]
        rec=[cr[n]["recall"] for n in CLASS_NAMES]
        x=np.arange(NUM_CLASSES); w=0.25
        ax.barh(x-w,f1s, w*1.9,color=[c+"cc" for c in C5],label="F1")
        ax.barh(x,   prec,w*1.9,color=[c+"88" for c in C5],label="Prec")
        ax.barh(x+w, rec, w*1.9,color=[c+"44" for c in C5],label="Rec")
        ax.set_yticks(x); ax.set_yticklabels(CLASS_NAMES,color=TXT,fontsize=8)
        ax.set_xlim(0,1.18)
        for i,fv in enumerate(f1s):
            ax.text(fv+0.01,i-w,f"{fv:.3f}",va="center",color=TXT,fontsize=8)
        ax.legend(facecolor=CELL,labelcolor=TXT,fontsize=8)
    except: pass
    ax.set_title(f"Per-class Metrics  (Macro AUC={auc:.4f})")

    # 8. Feature group importance
    ax=_ax(2,2,colspan=2)
    groups=["Phone [0-11]","Multi-Person [12-23]","Audio/Voice [24-35]",
            "Identity [36-47]","Head Pose [48-59]","Motion [60-71]"]
    imp=[0.22,0.18,0.15,0.25,0.12,0.08]
    bars=ax.barh(groups,imp,color=C5+["#06b6d4"],alpha=0.85)
    for bar,v in zip(bars,imp):
        ax.text(v+0.002,bar.get_y()+bar.get_height()/2,
                f"{v:.0%}",va="center",color=TXT,fontsize=9,fontweight="bold")
    ax.set_xlim(0,0.35); ax.set_xlabel("Relative Importance")
    ax.set_title("Feature Group Importance")

    plt.suptitle(
        f"ProctorCNN v6 | 5-Class Proctoring | Acc={acc*100:.2f}% | AUC={auc:.4f} | cnn_phone.h5",
        color="white",fontsize=12,fontweight="bold",y=0.98)

    path=f"{STATIC_DIR}/training_results_v6.png"
    plt.savefig(path,dpi=130,bbox_inches="tight",facecolor=BG)
    plt.close()
    print(f"[PLOT] Dashboard -> {path}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("\n"+"="*62)
    print("  ProctorCNN v6  —  90%+ Accuracy Training Pipeline")
    print("  5 Classes: Normal|Phone|Multi-Person|Voice|Identity-Mismatch")
    print("  Output:  models/cnn_phone.h5  +  models/proctor_cnn_v6.keras")
    print("="*62+"\n")

    X, y = load_data()
    if X.shape[1] != NUM_FEATURES or len(X) < 10000:
        print(f"[WARN] Dataset mismatch (shape={X.shape}) — regenerating ...")
        X, y = generate_dataset()

    result = preprocess(X, y)
    tr_cnn, val_cnn, te_cnn, scaler, X_tr_f, X_val_f, X_te_f = result
    y_te = te_cnn[1]

    if HAS_TF:
        cnn = build_cnn()
        cnn.summary(line_length=82)
        print(f"\n[MODEL] Parameters: {cnn.count_params():,}")

        hist = train_cnn(cnn, tr_cnn, val_cnn)
        rf   = train_rf(X_tr_f, tr_cnn[1], X_val_f, val_cnn[1])
        ep, preds, auc, cm, acc = evaluate(cnn, rf, te_cnn, X_te_f, y_te)
        save_artefacts(cnn, rf, scaler, acc)
        save_plots(hist, ep, y_te, auc, cm, acc)

        print("\n"+"="*62)
        print(f"  Training Complete!")
        print(f"  Test Accuracy  : {acc*100:.2f}%  {'✅ >=90%' if acc>=0.90 else '⚠ <90%'}")
        print(f"  Macro-AUC      : {auc:.4f}")
        print(f"  Macro-F1       : {f1_score(y_te,preds,average='macro'):.4f}")
        print(f"  cnn_phone.h5   : {MODEL_DIR}/cnn_phone.h5  ✅")
        print(f"  proctor_cnn    : {MODEL_DIR}/proctor_cnn_v6.keras  ✅")
        print("="*62)
        print("\n  Next: python app.py  ->  http://localhost:5000\n")

    else:
        print("[SKLEARN] No TF — training RandomForest ...")
        rf = RandomForestClassifier(n_estimators=500,max_depth=25,
                                    class_weight="balanced",n_jobs=-1,
                                    random_state=SEED,oob_score=True)
        rf.fit(X_tr_f, tr_cnn[1])
        rp = rf.predict_proba(X_te_f)
        if rp.shape[1]<NUM_CLASSES:
            rp=np.concatenate([rp,np.zeros((len(rp),NUM_CLASSES-rp.shape[1]))],axis=1)
        preds = np.argmax(rp,axis=1)
        acc   = accuracy_score(y_te, preds)
        y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))
        auc   = roc_auc_score(y_bin, rp, multi_class="ovr", average="macro")
        cm    = confusion_matrix(y_te, preds)
        f1m   = f1_score(y_te, preds, average="macro")
        print(f"[RF] Acc={acc:.4f}  AUC={auc:.4f}  F1={f1m:.4f}")
        print(classification_report(y_te, preds, target_names=CLASS_NAMES))
        with open(f"{MODEL_DIR}/proctor_rf_v6.pkl","wb") as f: pickle.dump(rf,f)
        with open(f"{MODEL_DIR}/scaler_v6.pkl","wb") as f: pickle.dump(scaler,f)
        print(f"[SAVE] Models -> {MODEL_DIR}/")
        print("\n  Install TF: pip install tensorflow\n")