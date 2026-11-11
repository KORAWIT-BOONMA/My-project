import pandas as pd
import os


# ตำแหน่งโฟลเดอร์ dataset
DATASET_DIR = "dataset"

# รายชื่อไฟล์
files = [
    "data_normal.csv",
    "data_rpm_fault.csv",
    "data_temp_fault.csv",
    "data_vib_fault.csv"
]


print("=" * 60)
print("ตรวจสอบ Dataset")
print("=" * 60)


for file in files:

    path = os.path.join(DATASET_DIR, file)

    print(f"\n📁 ไฟล์: {file}")

    # อ่าน CSV
    df = pd.read_csv(path)

    # จำนวนแถวและคอลัมน์
    print(f"จำนวนข้อมูล: {len(df)} rows")
    print(f"จำนวน columns: {len(df.columns)}")

    # แสดงชื่อ columns
    print("\nColumns:")
    print(df.columns.tolist())

    # ตรวจ missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # แสดงข้อมูล 5 แถวแรก
    print("\nตัวอย่างข้อมูล:")
    print(df.head())

    print("-" * 60)