import time
import cv2
from ultralytics import YOLO
from flask import Flask, Response

app = Flask(__name__)

# =========================
# 설정
# =========================

# 웹캠 인덱스
WEBCAM_INDEX = 0

# RTSP 주소
#RTSP_URL = "rtsp://sesac1234:sesac1234@172.16.8.25/stream1"
RTSP_URL = "cam01.mp4"

# auto / webcam / rtsp
SOURCE_MODE = "rtsp"

# YOLO 모델 로딩
MODEL_PATH = "yolo11n.pt"
print("[INFO] YOLO 모델 로딩 중...")
model = YOLO(MODEL_PATH)
print("[INFO] YOLO 모델 로딩 완료")

# 사람 클래스 ID (COCO 기준 0)
PERSON_CLASS_ID = 0


# =========================
#  웹캠 → 실패 → RTSP 자동 fallback
# =========================
def open_capture():
    if SOURCE_MODE == "webcam":
        print(f"[INFO] webcam only → {WEBCAM_INDEX}")
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if cap.isOpened():
            print("[INFO] 웹캠 열기 성공")
            return cap, f"webcam:{WEBCAM_INDEX}"
        print("[ERROR] 웹캠 열기 실패")
        return None, None

    if SOURCE_MODE == "rtsp":
        print(f"[INFO] rtsp only → {RTSP_URL}")
        cap = cv2.VideoCapture(RTSP_URL)
        if cap.isOpened():
            print("[INFO] RTSP 열기 성공")
            return cap, RTSP_URL
        print("[ERROR] RTSP 열기 실패")
        return None, None

    # auto 모드
    print("[INFO] SOURCE_MODE=auto → 웹캠 먼저 시도")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if cap.isOpened():
        print(f"[INFO] auto: 웹캠 성공 → {WEBCAM_INDEX}")
        return cap, f"webcam:{WEBCAM_INDEX}"

    print("[WARN] auto: 웹캠 실패 → RTSP 시도")

    cap = cv2.VideoCapture(RTSP_URL)
    if cap.isOpened():
        print(f"[INFO] auto: RTSP 성공 → {RTSP_URL}")
        return cap, RTSP_URL

    print("[FATAL] auto: 둘 다 실패")
    return None, None


# =========================
#  MJPEG Streaming + YOLO Tracking
# =========================
def generate():
    cap, source = open_capture()

    if cap is None:
        print("[ERROR] 영상 소스를 열지 못함")
        while True:
            # 빈 프레임 반복 송출 (UI는 검정 화면 유지)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"" + b"\r\n"
            )
            time.sleep(1)

    print(f"[INFO] 사용 중인 소스: {source}")

    fps_ema = 0.0

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패 → 재시도")
            time.sleep(0.05)
            continue

        # =========================
        # 사람(person)만 YOLO tracking
        # =========================
        results = model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS_ID],  # 🔥 사람만 추적
            verbose=False
        )

        # bbox + id + mask + label 모두 그린 이미지
        annotated = results[0].plot()

        # FPS
        t1 = time.time()
        inst_fps = 1.0 / max((t1 - t0), 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * inst_fps

        cv2.putText(
            annotated,
            f"FPS: {fps_ema:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        ok, jpeg = cv2.imencode(".jpg", annotated)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
