import os
import joblib
import pandas as pd


# ============================================================
# 1. กำหนด Path ของ Model
# ============================================================

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "motor_rf_model.pkl"
)


# ============================================================
# 2. กำหนด Feature ที่ใช้ในการ Prediction
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
# 3. ตรวจสอบว่ามี Model หรือไม่
# ============================================================

if not os.path.exists(MODEL_FILE):

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print("ไม่พบไฟล์ Model:")
    print(MODEL_FILE)

    print("\nกรุณาตรวจสอบว่าไฟล์")
    print("motor_rf_model.pkl")
    print("อยู่ในโฟลเดอร์ model หรือไม่")

    exit()


# ============================================================
# 4. โหลด Random Forest Model
# ============================================================

model = joblib.load(MODEL_FILE)


print("=" * 60)
print("Motor Health Prediction")
print("=" * 60)

print("โหลด Random Forest Model สำเร็จ")

print(f"Model: {MODEL_FILE}")


# ============================================================
# 5. รับค่าจากผู้ใช้
# ============================================================

print("\n" + "=" * 60)
print("กรอกค่าของมอเตอร์")
print("=" * 60)

print("หมายเหตุ: ตอนนี้เป็นการทดสอบด้วยการกรอกค่าเอง")
print()


try:

    rpm = float(input("RPM: "))

    temp = float(input("Temperature: "))

    vx = float(input("VX: "))

    vy = float(input("VY: "))

    vz = float(input("VZ: "))

    current = float(input("Current: "))

    pwm = float(input("PWM: "))


except ValueError:

    print("\nกรุณากรอกเฉพาะตัวเลข")

    exit()


# ============================================================
# 6. สร้าง DataFrame สำหรับ Prediction
# ============================================================
#
# ใช้ DataFrame เพื่อให้ชื่อ Feature ตรงกับตอน Train
# และป้องกัน Warning:
#
# X does not have valid feature names
#
# ============================================================

input_data = pd.DataFrame(
    [[
        rpm,
        temp,
        vx,
        vy,
        vz,
        current,
        pwm
    ]],
    columns=FEATURE_COLS
)


# ============================================================
# 7. แสดง Input ที่ส่งเข้า Model
# ============================================================

print("\n" + "=" * 60)
print("ข้อมูลที่ส่งเข้า Random Forest")
print("=" * 60)

print(input_data.to_string(index=False))


# ============================================================
# 8. Prediction
# ============================================================

prediction = model.predict(input_data)[0]


# ============================================================
# 9. แสดงผล Prediction
# ============================================================

print("\n" + "=" * 60)
print("ผลการทำนาย")
print("=" * 60)

print(f"Motor Status: {prediction}")


# ============================================================
# 10. แสดงความน่าจะเป็นของแต่ละ Class
# ============================================================

probabilities = model.predict_proba(input_data)[0]


print("\n" + "=" * 60)
print("ความน่าจะเป็นของแต่ละ Class")
print("=" * 60)


for class_name, probability in zip(
    model.classes_,
    probabilities
):

    print(
        f"{class_name:12s}: "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# 11. แสดง Class ที่มีความน่าจะเป็นสูงสุด
# ============================================================

max_probability = probabilities.max() * 100


print("\n" + "=" * 60)
print("สรุปผล")
print("=" * 60)

print(f"สถานะมอเตอร์ : {prediction}")
print(f"Confidence   : {max_probability:.2f}%")


print("\n" + "=" * 60)
print("Prediction เสร็จสิ้น")
print("=" * 60)