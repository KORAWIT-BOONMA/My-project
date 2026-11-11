import os
import json
import joblib
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "motor_rf_model.pkl"
)


# ============================================================
# MQTT CONFIG
# ============================================================

MQTT_BROKER = "10.38.12.151"
MQTT_PORT = 1883
MQTT_TOPIC = "motor/sensors"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "rpm",
    "temp",
    "vx",
    "vy",
    "vz",
    "current",
    "pwm"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("MOTOR HEALTH - MQTT RANDOM FOREST PREDICTOR")
print("=" * 70)

print("\nกำลังโหลด Model...")

if not os.path.exists(MODEL_PATH):

    print("\nERROR: ไม่พบ Model")
    print(MODEL_PATH)
    exit()


model = joblib.load(MODEL_PATH)

print("โหลด Model สำเร็จ")

print(
    f"Model: {type(model).__name__}"
)

print(
    f"Features: {FEATURES}"
)


# ============================================================
# SHOW CLASSES
# ============================================================

print("\nClass ที่สามารถทำนาย:")

for class_name in model.classes_:

    print(f"- {class_name}")


# ============================================================
# CONVERT MQTT DATA
# ============================================================

def convert_data(data):

    """
    แปลงข้อมูลจาก MQTT JSON
    ให้ตรงกับ Feature ที่ Random Forest ใช้
    """

    # --------------------------------------------------------
    # RPM
    # --------------------------------------------------------

    if "rpm" not in data:

        raise ValueError(
            "ไม่มีค่า rpm"
        )

    rpm = data["rpm"]


    # --------------------------------------------------------
    # Temperature
    # รองรับ temp และ temperature
    # --------------------------------------------------------

    if "temp" in data:

        temp = data["temp"]

    elif "temperature" in data:

        temp = data["temperature"]

    else:

        raise ValueError(
            "ไม่มีค่า temp หรือ temperature"
        )


    # --------------------------------------------------------
    # Vibration X
    # --------------------------------------------------------

    if "vx" in data:

        vx = data["vx"]

    elif "vibration_x" in data:

        vx = data["vibration_x"]

    else:

        raise ValueError(
            "ไม่มีค่า vx หรือ vibration_x"
        )


    # --------------------------------------------------------
    # Vibration Y
    # --------------------------------------------------------

    if "vy" in data:

        vy = data["vy"]

    elif "vibration_y" in data:

        vy = data["vibration_y"]

    else:

        raise ValueError(
            "ไม่มีค่า vy หรือ vibration_y"
        )


    # --------------------------------------------------------
    # Vibration Z
    # --------------------------------------------------------

    if "vz" in data:

        vz = data["vz"]

    elif "vibration_z" in data:

        vz = data["vibration_z"]

    else:

        raise ValueError(
            "ไม่มีค่า vz หรือ vibration_z"
        )


    # --------------------------------------------------------
    # Current
    # --------------------------------------------------------

    if "current" not in data:

        raise ValueError(
            "ไม่มีค่า current"
        )

    current = data["current"]


    # --------------------------------------------------------
    # PWM
    # --------------------------------------------------------

    if "pwm" not in data:

        raise ValueError(
            "ไม่มีค่า pwm"
        )

    pwm = data["pwm"]


    # --------------------------------------------------------
    # Convert เป็นตัวเลข
    # --------------------------------------------------------

    result = {

        "rpm": float(rpm),

        "temp": float(temp),

        "vx": float(vx),

        "vy": float(vy),

        "vz": float(vz),

        "current": float(current),

        "pwm": float(pwm)
    }


    return result


# ============================================================
# PREDICTION
# ============================================================

def predict_motor(data):

    try:

        # ----------------------------------------------------
        # แปลงข้อมูล
        # ----------------------------------------------------

        converted = convert_data(data)


        # ----------------------------------------------------
        # สร้าง DataFrame
        #
        # สำคัญ:
        # ใช้ชื่อ Feature ให้ตรงกับตอน Train
        # ----------------------------------------------------

        X = pd.DataFrame(
            [[
                converted["rpm"],
                converted["temp"],
                converted["vx"],
                converted["vy"],
                converted["vz"],
                converted["current"],
                converted["pwm"]
            ]],
            columns=FEATURES
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(X)

        predicted_class = prediction[0]


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(X)[0]


        # ----------------------------------------------------
        # Probability ของแต่ละ Class
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("ความน่าจะเป็นของแต่ละ Class")
        print("=" * 70)

        for class_name, probability in zip(
            model.classes_,
            probabilities
        ):

            print(
                f"{class_name:<12}: "
                f"{probability * 100:.2f}%"
            )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = np.max(
            probabilities
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("ผลการทำนาย")
        print("=" * 70)

        print(
            f"สถานะมอเตอร์ : {predicted_class}"
        )

        print(
            f"Confidence    : "
            f"{confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # แสดงค่าที่ส่งเข้า Model
        # ----------------------------------------------------

        print("\nค่าที่ส่งเข้า Random Forest")
        print("-" * 70)

        print(
            f"rpm     : {converted['rpm']}"
        )

        print(
            f"temp    : {converted['temp']}"
        )

        print(
            f"vx      : {converted['vx']}"
        )

        print(
            f"vy      : {converted['vy']}"
        )

        print(
            f"vz      : {converted['vz']}"
        )

        print(
            f"current : {converted['current']}"
        )

        print(
            f"pwm     : {converted['pwm']}"
        )

        print("=" * 70)


    except Exception as e:

        print("\nERROR: ไม่สามารถทำนายได้")

        print(
            f"รายละเอียด: {e}"
        )


# ============================================================
# MQTT CONNECT CALLBACK
# ============================================================

def on_connect(
    client,
    userdata,
    flags,
    rc,
    properties=None
):

    if rc == 0:

        print("\n" + "=" * 70)
        print("MQTT CONNECTED")
        print("=" * 70)

        print(
            f"Broker : {MQTT_BROKER}"
        )

        print(
            f"Port   : {MQTT_PORT}"
        )

        print(
            f"Topic  : {MQTT_TOPIC}"
        )

        print("\nกำลังรอข้อมูลจาก Motor...")


        client.subscribe(
            MQTT_TOPIC
        )


    else:

        print(
            f"\nMQTT Connection Failed: {rc}"
        )


# ============================================================
# MQTT MESSAGE CALLBACK
# ============================================================

def on_message(
    client,
    userdata,
    msg
):

    print("\n\n" + "=" * 70)
    print("ได้รับ MQTT Message")
    print("=" * 70)


    print(
        f"Topic: {msg.topic}"
    )


    # --------------------------------------------------------
    # Decode JSON
    # --------------------------------------------------------

    try:

        payload = msg.payload.decode(
            "utf-8"
        )

        print("\nPayload:")
        print(payload)


        data = json.loads(
            payload
        )


    except Exception as e:

        print("\nERROR: JSON ไม่ถูกต้อง")
        print(e)

        return


    # --------------------------------------------------------
    # แสดงข้อมูลที่ได้รับ
    # --------------------------------------------------------

    print("\nข้อมูลที่ได้รับ:")

    for key, value in data.items():

        print(
            f"{key:<15}: {value}"
        )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predict_motor(data)


# ============================================================
# MQTT CLIENT
# ============================================================

client = mqtt.Client(
    callback_api_version=
    mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect = on_connect

client.on_message = on_message


# ============================================================
# CONNECT MQTT
# ============================================================

print("\n" + "=" * 70)
print("กำลังเชื่อมต่อ MQTT...")
print("=" * 70)

print(
    f"Broker : {MQTT_BROKER}"
)

print(
    f"Port   : {MQTT_PORT}"
)

print(
    f"Topic  : {MQTT_TOPIC}"
)


try:

    client.connect(
        MQTT_BROKER,
        MQTT_PORT,
        60
    )


except Exception as e:

    print("\nไม่สามารถเชื่อมต่อ MQTT ได้")

    print(e)

    exit()


# ============================================================
# LOOP
# ============================================================

print("\nเริ่ม MQTT Loop...")

client.loop_forever()