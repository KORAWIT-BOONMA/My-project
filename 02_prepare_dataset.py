import pandas as pd
import os


# ============================================================
# ตั้งค่าที่อยู่ Dataset
# ============================================================

DATASET_DIR = "dataset"


# ============================================================
# กำหนดไฟล์และ Label
# ============================================================

DATASETS = {
    "data_normal.csv": "normal",
    "data_rpm_fault.csv": "rpm_fault",
    "data_temp_fault.csv": "temp_fault",
    "data_vib_fault.csv": "vib_fault"
}


# ============================================================
# Feature ที่จะใช้ในการ Train AI
# ============================================================

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
# อ่าน Dataset แต่ละไฟล์
# ============================================================

dataframes = []

print("=" * 60)
print("เตรียม Dataset สำหรับ Random Forest")
print("=" * 60)


for filename, label in DATASETS.items():

    path = os.path.join(DATASET_DIR, filename)

    print(f"\nกำลังอ่าน: {filename}")

    # อ่าน CSV
    df = pd.read_csv(path)

    # ตรวจสอบ Feature
    missing_features = [
        col for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing_features:
        print("ERROR: ไม่พบ columns:")
        print(missing_features)
        exit()

    # เพิ่ม Label
    df["label"] = label

    # เก็บ DataFrame ไว้
    dataframes.append(df)

    print(f"จำนวนข้อมูล: {len(df)}")
    print(f"Label: {label}")


# ============================================================
# รวมข้อมูลในหน่วยความจำ
# ============================================================

df_all = pd.concat(dataframes, ignore_index=True)


# ============================================================
# แสดงผลจำนวนข้อมูลแต่ละ Class
# ============================================================

print("\n" + "=" * 60)
print("ผลรวม Dataset")
print("=" * 60)

print(f"จำนวนข้อมูลทั้งหมด: {len(df_all)} rows")

print("\nจำนวนข้อมูลแต่ละ Class:")
print(df_all["label"].value_counts())


# ============================================================
# สร้าง X และ y
# ============================================================

X = df_all[FEATURE_COLS]

y = df_all["label"]


print("\n" + "=" * 60)
print("X และ y")
print("=" * 60)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\nFeature ที่ใช้:")
print(FEATURE_COLS)

print("\nLabel ที่มี:")
print(y.unique())


# ============================================================
# แสดงตัวอย่างข้อมูล
# ============================================================

print("\n" + "=" * 60)
print("ตัวอย่างข้อมูล")
print("=" * 60)

print(df_all.head())


print("\nเตรียม Dataset เสร็จเรียบร้อยแล้ว")