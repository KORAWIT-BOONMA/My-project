import os
import joblib


# ============================================================
# 1. กำหนดตำแหน่ง Model
# ============================================================

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model",
    "motor_rf_model.pkl"
)


# ============================================================
# 2. ตรวจสอบว่า Model มีอยู่หรือไม่
# ============================================================

if not os.path.exists(MODEL_FILE):

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print("ไม่พบไฟล์ Model:")
    print(MODEL_FILE)

    exit()


# ============================================================
# 3. โหลด Model
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# 4. แสดงข้อมูล Model
# ============================================================

print("=" * 60)
print("RANDOM FOREST MODEL INFORMATION")
print("=" * 60)

print("\nModel type:")
print(type(model).__name__)


# ============================================================
# 5. จำนวน Decision Trees
# ============================================================

print("\nจำนวน Decision Trees:")
print(model.n_estimators)


# ============================================================
# 6. Feature ที่ Model ใช้
# ============================================================

print("\nFeature ที่ Model ใช้:")

if hasattr(model, "feature_names_in_"):

    for i, feature in enumerate(
        model.feature_names_in_,
        start=1
    ):
        print(f"{i}. {feature}")

else:

    print("ไม่พบชื่อ Feature ใน Model")


# ============================================================
# 7. Class ที่ Model สามารถทำนาย
# ============================================================

print("\nClass ที่ Model สามารถทำนาย:")

for i, class_name in enumerate(
    model.classes_,
    start=1
):

    print(f"{i}. {class_name}")


# ============================================================
# 8. Feature Importance
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


if hasattr(model, "feature_importances_"):

    features = model.feature_names_in_
    importances = model.feature_importances_

    feature_data = list(
        zip(features, importances)
    )

    feature_data.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in feature_data:

        print(
            f"{feature:10s} : "
            f"{importance:.6f} "
            f"({importance * 100:.2f}%)"
        )


# ============================================================
# 9. ตรวจสอบจำนวน Feature
# ============================================================

print("\n" + "=" * 60)
print("ตรวจสอบ Feature")
print("=" * 60)

print(
    f"จำนวน Feature ที่ Model รับ: "
    f"{model.n_features_in_}"
)


# ============================================================
# 10. สรุป
# ============================================================

print("\n" + "=" * 60)
print("สรุป Model")
print("=" * 60)

print(f"Model file : {MODEL_FILE}")

print(
    f"Algorithm  : "
    f"Random Forest"
)

print(
    f"Trees      : "
    f"{model.n_estimators}"
)

print(
    f"Features   : "
    f"{model.n_features_in_}"
)

print(
    f"Classes    : "
    f"{len(model.classes_)}"
)

print("\nModel พร้อมสำหรับขั้นตอน Prediction")
print("=" * 60)