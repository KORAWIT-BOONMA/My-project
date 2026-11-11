import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ใช้ Dataset ใหม่
DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset_new"
)

# ที่เก็บ Model
MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "motor_rf_model.pkl"
)


# ============================================================
# DATASET
# ============================================================

DATASETS = {
    "normal": "data_normal.csv",
    "rpm_fault": "data_rpm_fault.csv",
    "temp_fault": "data_temp_fault.csv",
    "vib_fault": "data_vib_fault.csv"
}


# Feature ที่ใช้ Train
FEATURE_COLS = [
    "rpm",
    "temp",
    "vx",
    "vy",
    "vz",
    "current",
    "pwm"
]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RANDOM FOREST - FULL DATASET TRAINING")
print("=" * 70)

print()
print("Dataset ที่ใช้:")
print(DATASET_DIR)

print()
print("จะใช้ข้อมูลทั้งหมดในการ Train")
print("ไม่มีการแบ่ง Train / Test")

print()
print("กำลังโหลด Dataset...")
print("-" * 70)


# ============================================================
# LOAD DATA
# ============================================================

all_data = []

for label, filename in DATASETS.items():

    path = os.path.join(
        DATASET_DIR,
        filename
    )

    if not os.path.exists(path):

        print()
        print("ไม่พบไฟล์:")
        print(path)
        print()

        print("ตรวจสอบว่าไฟล์อยู่ใน:")
        print(DATASET_DIR)

        raise SystemExit

    df = pd.read_csv(path)

    print(
        f"{filename:<25} "
        f"{len(df):>5} samples"
    )

    # กำหนด Label จากชื่อไฟล์
    df["label"] = label

    all_data.append(df)


# ============================================================
# COMBINE DATA
# ============================================================

df = pd.concat(
    all_data,
    ignore_index=True
)


print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print()
print(f"จำนวนข้อมูลทั้งหมด: {len(df)} samples")

print()
print("จำนวนแต่ละ Class:")

print(
    df["label"].value_counts()
)


# ============================================================
# CHECK FEATURE
# ============================================================

print()
print("=" * 70)
print("ตรวจสอบ Feature")
print("=" * 70)

missing_features = [
    col
    for col in FEATURE_COLS
    if col not in df.columns
]

if missing_features:

    print()
    print("พบ Feature ที่ขาด:")
    print(missing_features)

    raise SystemExit


print()
print("Feature ที่ใช้:")

for feature in FEATURE_COLS:
    print("-", feature)


# ============================================================
# X / y
# ============================================================

X = df[FEATURE_COLS]

y = df["label"]


print()
print("=" * 70)
print("X / y")
print("=" * 70)

print()
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("กำลัง Train Random Forest...")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X,
    y
)

print()
print("Train เสร็จแล้ว!")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = model.feature_importances_

for feature, value in sorted(
    zip(FEATURE_COLS, importance),
    key=lambda x: x[1],
    reverse=True
):

    print(
        f"{feature:<10}: "
        f"{value:.6f} "
        f"({value * 100:.2f}%)"
    )


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)


print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print()
print("Model file:")
print(MODEL_PATH)

print()
print("Algorithm : Random Forest")
print("Trees     :", model.n_estimators)
print("Samples   :", len(df))
print("Features  :", len(FEATURE_COLS))
print("Classes   :", len(model.classes_))

print()
print("Class ที่ Model เรียนรู้:")

for class_name in model.classes_:
    print("-", class_name)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)