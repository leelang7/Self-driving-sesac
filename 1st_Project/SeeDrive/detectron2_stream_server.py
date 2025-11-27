import cv2
import numpy as np
import math
from dataclasses import dataclass
import threading
import time
from flask import Flask, Response, request, jsonify

# =========================
# 설정
# =========================
USE_DUMMY_SLOTS = False  # True면 더미 슬롯, False면 Detectron2 사용
VIDEO_SOURCE = r"/home/elicer/junlee/back_parking/videos/front1.mov"  # 웹캠, 동영상 파일, 네트워크카메라 주소

FRAME_W = 1280
FRAME_H = 720

# Detectron2 설정 (환경에 맞게 수정)
DETECTRON_CFG_FILE = "detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
DETECTRON_WEIGHTS  = r"/home/elicer/sechan/trained_output/output/DT_cls4_1008x760_iter20000/model_final.pth"
DETECTRON_DEVICE   = "cuda"   # "cuda" 또는 "cpu"

PARKING_CLASS_ID = 1   # Parking Area 클래스 ID
MIN_SLOT_AREA    = 200 # 너무 작은 마스크는 무시


# =========================
# 데이터 구조
# =========================
@dataclass
class ParkingSlot:
    slot_id: int
    polygon: np.ndarray  # (N,2) int32
    center: tuple        # (x, y)


@dataclass
class CarPose:
    x: float
    y: float
    yaw: float           # 라디안
    reverse: bool = False  # 후진 구간 여부


def get_initial_car_pose() -> CarPose:
    """항상 내 시점(화면 아래 중앙, 위쪽을 바라보는 자세)에서 시작"""
    return CarPose(FRAME_W // 2, int(FRAME_H * 0.9), -math.pi / 2)


# =========================
# Detectron2 ParkingSlot Detector
# =========================
class ParkingDetector:
    def __init__(self, use_dummy=True):
        self.use_dummy = use_dummy
        self.slot_id_counter = 0

        if not use_dummy:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor

            cfg = get_cfg()
            cfg.merge_from_file(DETECTRON_CFG_FILE)

            # 학습 때와 동일한 클래스 개수로 맞추기
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4

            cfg.MODEL.WEIGHTS = DETECTRON_WEIGHTS
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = DETECTRON_DEVICE

            self.predictor = DefaultPredictor(cfg)
        else:
            self.predictor = None

    def detect(self, frame):
        if self.use_dummy:
            return self._dummy_slots(frame)
        return self._detectron_slots(frame)

    def _dummy_slots(self, frame):
        """테스트용 더미 슬롯."""
        h, w, _ = frame.shape
        rects = [
            [(int(w * 0.60), int(h * 0.60)), (int(w * 0.80), int(h * 0.70))],
            [(int(w * 0.20), int(h * 0.50)), (int(w * 0.40), int(h * 0.60))],
            [(int(w * 0.45), int(h * 0.40)), (int(w * 0.65), int(h * 0.50))],
        ]
        slots = []
        self.slot_id_counter = 0
        for (x1, y1), (x2, y2) in rects:
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            self.slot_id_counter += 1
            slots.append(ParkingSlot(self.slot_id_counter, poly, (cx, cy)))
        return slots

    def _detectron_slots(self, frame):
        """
        Detectron2 InstanceSegmentation 마스크에서 Parking Area만 추출.
        """
        outputs = self.predictor(frame)
        instances = outputs["instances"].to("cpu")

        if not instances.has("pred_masks"):
            return []

        masks = instances.pred_masks.numpy()        # (N, H, W) bool
        classes = instances.pred_classes.numpy()    # (N,)

        slots = []
        self.slot_id_counter = 0

        for mask, cid in zip(masks, classes):
            if PARKING_CLASS_ID is not None and cid != PARKING_CLASS_ID:
                continue

            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < MIN_SLOT_AREA:
                continue

            poly = cnt.reshape(-1, 2)  # (N, 1, 2) -> (N, 2)
            poly = poly.astype(np.int32)

            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())

            self.slot_id_counter += 1
            slots.append(ParkingSlot(self.slot_id_counter, poly, (cx, cy)))

        return slots


# =========================
# Slot Selector (웹에서 클릭 좌표 전달)
# =========================
class SlotSelector:
    def __init__(self):
        self.slots = []
        self.selected_slot_id = None

    def update(self, slots):
        self.slots = slots

    def click(self, x, y):
        """웹에서 받은 클릭 좌표 처리"""
        clicked = False
        for slot in self.slots:
            if cv2.pointPolygonTest(slot.polygon, (x, y), False) >= 0:
                self.selected_slot_id = slot.slot_id
                clicked = True
                print("[INFO] 슬롯 선택:", slot.slot_id)
                break
        if not clicked:
            print("[INFO] 빈 공간 클릭 → 선택 해제")
            self.selected_slot_id = None

    def get(self):
        if self.selected_slot_id is None:
            return None
        for s in self.slots:
            if s.slot_id == self.selected_slot_id:
                return s
        return None

    def clear(self):
        self.selected_slot_id = None


# =========================
# Parking Planner (원본 그대로)
# =========================
class ParkingPlanner:
    def __init__(self):
        self.path = []

    def plan(self, start_pose: CarPose, slot: ParkingSlot):
        mid_x = FRAME_W / 2
        if slot.center[0] <= mid_x:
            self.path = self._plan_left(start_pose, slot)
            return self.path
        else:
            start_m = self._mirror_pose(start_pose)
            slot_m = self._mirror_slot(slot)
            path_m = self._plan_left(start_m, slot_m)
            self.path = [self._mirror_pose(p) for p in path_m]
            return self.path

    @staticmethod
    def _mirror_pose(p: CarPose) -> CarPose:
        x_m = FRAME_W - p.x
        y_m = p.y
        yaw_m = math.pi - p.yaw
        yaw_m = ((yaw_m + math.pi) % (2 * math.pi)) - math.pi
        return CarPose(x_m, y_m, yaw_m, reverse=p.reverse)

    @staticmethod
    def _mirror_slot(s: ParkingSlot) -> ParkingSlot:
        poly = s.polygon.astype(float)
        poly[:, 0] = FRAME_W - poly[:, 0]
        cx = FRAME_W - s.center[0]
        cy = s.center[1]
        return ParkingSlot(s.slot_id, poly.astype(np.int32), (int(cx), int(cy)))

    def _plan_left(self, start_pose: CarPose, slot: ParkingSlot):
        cx, cy = slot.center
        pts = slot.polygon.reshape(-1, 2).astype(np.float32)
        mean = pts.mean(axis=0, keepdims=True)
        pts_c = pts - mean
        cov = np.cov(pts_c.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        main_vec = eigvecs[:, np.argmax(eigvals)]
        slot_yaw = math.atan2(main_vec[1], main_vec[0])
        slot_yaw = ((slot_yaw + math.pi) % (2 * math.pi)) - math.pi

        park_pose = CarPose(cx, cy, slot_yaw, reverse=True)

        lane_yaw = start_pose.yaw
        lane_dir = np.array([math.cos(lane_yaw), math.sin(lane_yaw)])
        vec_to_slot = np.array([cx - start_pose.x, cy - start_pose.y], dtype=np.float32)
        proj_dist = float(np.dot(vec_to_slot, lane_dir))
        if proj_dist < 0:
            proj_dist = 0.0

        margin = 120.0
        forward_dist = proj_dist + margin

        fx = start_pose.x + lane_dir[0] * forward_dist
        fy = start_pose.y + lane_dir[1] * forward_dist
        lane_forward_pose = CarPose(fx, fy, lane_yaw, reverse=False)

        forward_path = self._straight_forward(start_pose, lane_forward_pose, steps=60)

        reverse_path = self._reverse_curve_then_straight_from_lane(
            lane_forward_pose,
            park_pose,
            slot_yaw,
            slot.center,
            total_steps=110,
        )

        return forward_path + reverse_path

    def _straight_forward(self, p0: CarPose, p1: CarPose, steps=60):
        path = []
        for i in range(steps):
            t = i / (steps - 1)
            x = p0.x * (1 - t) + p1.x * t
            y = p0.y * (1 - t) + p1.y * t
            yaw = p0.yaw
            path.append(CarPose(x, y, yaw, reverse=False))
        return path

    def _reverse_curve_then_straight_from_lane(
        self,
        start_pose: CarPose,
        park_pose: CarPose,
        slot_yaw: float,
        slot_center: tuple,
        total_steps=110,
    ):
        curve_ratio = 0.6
        curve_steps = max(2, int(total_steps * curve_ratio))
        straight_steps = max(1, total_steps - curve_steps)

        front_offset = 190.0
        mx = park_pose.x + front_offset * math.cos(slot_yaw)
        my = park_pose.y + front_offset * math.sin(slot_yaw)
        mid_pose = CarPose(mx, my, slot_yaw, reverse=True)

        p0v = np.array([start_pose.x, start_pose.y], dtype=np.float32)
        p2v = np.array([mid_pose.x, mid_pose.y], dtype=np.float32)
        scv = np.array([slot_center[0], slot_center[1]], dtype=np.float32)

        base = p2v - p0v
        base_norm = np.linalg.norm(base)
        if base_norm < 1e-5:
            base = np.array([1.0, 0.0], dtype=np.float32)
            base_norm = 1.0
        base_dir = base / base_norm

        to_slot = scv - p0v
        cross = base_dir[0] * to_slot[1] - base_dir[1] * to_slot[0]
        left_normal = np.array([-base_dir[1], base_dir[0]])

        curve_side_offset = 80.0
        if cross >= 0:
            shift = -left_normal * curve_side_offset
        else:
            shift = left_normal * curve_side_offset

        mid_line = 0.5 * (p0v + p2v)
        p1v = mid_line + shift

        bezier_positions = []
        for i in range(curve_steps):
            t = i / (curve_steps - 1)
            one_t = 1.0 - t
            pos = (one_t * one_t) * p0v + 2.0 * one_t * t * p1v + (t * t) * p2v
            bezier_positions.append(pos)

        path = []
        lane_yaw = start_pose.yaw
        for i in range(curve_steps):
            t = i / (curve_steps - 1)
            x, y = bezier_positions[i]
            yaw = self._smooth_yaw(lane_yaw, slot_yaw, t)
            path.append(CarPose(float(x), float(y), yaw, reverse=True))

        sx, sy = path[-1].x, path[-1].y
        for i in range(1, straight_steps + 1):
            t = i / straight_steps
            x = sx * (1 - t) + park_pose.x * t
            y = sy * (1 - t) + park_pose.y * t
            yaw = slot_yaw
            path.append(CarPose(x, y, yaw, reverse=True))

        return path

    @staticmethod
    def _smooth_yaw(y0, y1, t):
        t2 = t * t * (3 - 2 * t)
        dy = ((y1 - y0 + math.pi) % (2 * math.pi)) - math.pi
        return y0 + dy * t2


# =========================
# Path Follower
# =========================
class PathFollower:
    def __init__(self):
        self.path = []
        self.idx = 0
        self.active = False

    def start(self, path):
        self.path = path
        self.idx = 0
        self.active = True

    def step(self):
        if not self.active or not self.path:
            return None
        if self.idx >= len(self.path):
            self.active = False
            return self.path[-1]
        pose = self.path[self.idx]
        self.idx += 1
        return pose


# =========================
# BEV Renderer
# =========================
class BEVRenderer:
    def __init__(self, w=600, h=600):
        self.w = w
        self.h = h

    def _draw_car(self, img, p: CarPose):
        car_len = 40
        car_wid = 20
        sx = self.w / FRAME_W
        sy = self.h / FRAME_H
        cx = int(p.x * sx)
        cy = int(p.y * sy)

        rect = np.array([
            [-car_len / 2, -car_wid / 2],
            [car_len / 2, -car_wid / 2],
            [car_len / 2, car_wid / 2],
            [-car_len / 2, car_wid / 2],
        ], dtype=np.float32)

        R = np.array([
            [math.cos(p.yaw), -math.sin(p.yaw)],
            [math.sin(p.yaw),  math.cos(p.yaw)],
        ])
        rect = (R @ rect.T).T
        rect[:, 0] += cx
        rect[:, 1] += cy
        rect = rect.astype(np.int32)

        color = (0, 0, 255) if not p.reverse else (0, 255, 255)
        cv2.polylines(img, [rect], True, color, 2)

    def _draw_slots(self, img, slots):
        sx = self.w / FRAME_W
        sy = self.h / FRAME_H
        for s in slots:
            poly = s.polygon.astype(np.float32)
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            poly = poly.astype(np.int32)
            cv2.polylines(img, [poly], True, (0, 255, 0), 2)

    def _draw_path(self, img, path):
        if not path:
            return
        sx = self.w / FRAME_W
        sy = self.h / FRAME_H
        pts = []
        for p in path:
            pts.append([int(p.x * sx), int(p.y * sy)])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], False, (255, 255, 255), 1)

    def render(self, slots, car_pose, path):
        bev = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_slots(bev, slots)
        self._draw_path(bev, path)
        if car_pose is not None:
            self._draw_car(bev, car_pose)
        cv2.putText(bev, "BEV", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return bev


# =========================
# FrontView Car 렌더링
# =========================
def draw_car_front(img, p: CarPose):
    car_len = 70
    car_wid = 35
    cx = int(p.x)
    cy = int(p.y)

    rect = np.array([
        [-car_len / 2, -car_wid / 2],
        [car_len / 2, -car_wid / 2],
        [car_len / 2, car_wid / 2],
        [-car_len / 2, car_wid / 2],
    ], dtype=np.float32)

    R = np.array([
        [math.cos(p.yaw), -math.sin(p.yaw)],
        [math.sin(p.yaw),  math.cos(p.yaw)],
    ])
    rect = (R @ rect.T).T
    rect[:, 0] += cx
    rect[:, 1] += cy
    rect = rect.astype(np.int32)

    color = (0, 0, 255) if not p.reverse else (0, 255, 255)
    cv2.polylines(img, [rect], True, color, 3)

    if not p.reverse:
        front = rect[1]
    else:
        front = rect[3]
    cv2.circle(img, tuple(front), 6, (0, 0, 255), -1)


# =========================
# 전역 상태 (스트리밍용)
# =========================
app = Flask(__name__)

state_lock = threading.Lock()

detector = ParkingDetector(USE_DUMMY_SLOTS)
selector = SlotSelector()
planner = ParkingPlanner()
follower = PathFollower()
bev_renderer = BEVRenderer()

car_pose = get_initial_car_pose()

freeze = False
freeze_frame = None
freeze_slots = []
slots = []
path = []
latest_raw_frame = None
front_view_frame = None
bev_frame = None

pending_click = None   # (x, y)
pending_reset = False


# =========================
# 백그라운드: Detectron2 + Parking 로직
# =========================
def processing_loop():
    global car_pose, freeze, freeze_frame, freeze_slots
    global slots, path, latest_raw_frame, front_view_frame, bev_frame
    global pending_click, pending_reset

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    while True:
        with state_lock:
            if not freeze:
                ret, frame = cap.read()
                if not ret:
                    # 입력이 끊기면 조금 쉬었다가 계속 시도
                    time.sleep(0.05)
                    continue
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                latest_raw_frame = frame.copy()

                slots = detector.detect(frame)
                selector.update(slots)
                current = frame.copy()
                slots_to_draw = slots
                mode_text = "SEARCH (click slot via /click, /reset to cancel)"
            else:
                if freeze_frame is None:
                    # 이론상 없어야 하지만 안전장치
                    freeze = False
                    continue
                current = freeze_frame.copy()
                slots_to_draw = freeze_slots
                mode_text = "PARKING (use /reset to cancel)"

            # 클릭 처리 (마우스 이벤트 대체)
            if pending_click is not None:
                x, y = pending_click
                pending_click = None
                selector.click(x, y)
                selected = selector.get()
                if selected is not None and not freeze:
                    print("[INFO] Freeze ON & Path Planning")
                    freeze = True
                    freeze_frame = current.copy()
                    freeze_slots = slots_to_draw.copy()

                    start_pose = get_initial_car_pose()
                    car_pose = start_pose

                    path = planner.plan(start_pose, selected)
                    follower.start(path)

            # reset 요청
            if pending_reset:
                print("[INFO] Reset / Cancel parking (HTTP)")
                pending_reset = False
                freeze = False
                selector.clear()
                follower.active = False
                path = []
                car_pose = get_initial_car_pose()

            # 경로 추종
            if follower.active:
                new_pose = follower.step()
                if new_pose is not None:
                    car_pose = new_pose

            # FrontView 렌더링
            overlay = current.copy()
            selected = selector.get()
            for s in slots_to_draw:
                col = (0, 255, 0)
                if selected is not None and s.slot_id == selected.slot_id:
                    col = (0, 255, 255)
                cv2.polylines(overlay, [s.polygon], True, col, 2)
                cx, cy = s.center
                cv2.putText(overlay, str(s.slot_id), (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

            draw_car_front(overlay, car_pose)
            cv2.putText(overlay, mode_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)

            front_view_frame = overlay
            bev_frame = bev_renderer.render(slots_to_draw, car_pose, path)

        time.sleep(0.02)  # 약 50fps 정도 타겟


# =========================
# MJPEG 스트림 generator
# =========================
def mjpeg_generator(frame_name: str):
    global front_view_frame, bev_frame
    while True:
        with state_lock:
            frame = front_view_frame if frame_name == "front" else bev_frame
            if frame is None:
                continue
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )
        time.sleep(0.03)


# =========================
# Flask 라우트
# =========================
@app.route("/front")
def stream_front():
    return Response(
        mjpeg_generator("front"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/bev")
def stream_bev():
    return Response(
        mjpeg_generator("bev"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/click", methods=["POST"])
def click():
    global pending_click
    data = request.get_json(force=True)
    x = int(data.get("x", 0))
    y = int(data.get("y", 0))
    print("[HTTP CLICK]", x, y)   # <-- 디버그용
    with state_lock:
        pending_click = (x, y)
    return jsonify({"status": "ok", "x": x, "y": y})



@app.route("/reset", methods=["POST"])
def reset():
    global pending_reset
    with state_lock:
        pending_reset = True
    return jsonify({"status": "ok"})


@app.route("/")
@app.route("/")
def index():
    # 간단한 HTML: FrontView/BEV 두 개 띄우고 클릭 보내기
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Parking Demo (Detectron2)</title>
  <style>
    body { background:#222; color:#eee; font-family:sans-serif; }
    .row { display:flex; gap:10px; }
    img { border:1px solid #555; }
    button { margin-top:10px; padding:6px 12px; }
  </style>
</head>
<body>
  <h2>Parking Demo (FrontView / BEV)</h2>
  <div class="row">
    <div>
      <p>FrontView (여기를 클릭해서 슬롯 선택)</p>
      <img id="front" src="/front" width="640" height="360" />
    </div>
    <div>
      <p>BEV</p>
      <img id="bev" src="/bev" width="600" height="600" />
    </div>
  </div>
  <button onclick="resetParking()">Reset</button>

  <script>
    // 서버 쪽 프레임 해상도와 맞춰줘야 함
    const FRAME_W = 1280;
    const FRAME_H = 720;
    const frontImg = document.getElementById('front');

    frontImg.addEventListener('click', async (e) => {
      const rect = frontImg.getBoundingClientRect();
      // img 태그는 640x360으로 표시되지만, 실제 프레임은 1280x720이므로 스케일링
      const x = Math.round((e.clientX - rect.left) * (FRAME_W / rect.width));
      const y = Math.round((e.clientY - rect.top)  * (FRAME_H / rect.height));

      console.log("click:", x, y);

      try {
        const res = await fetch('/click', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({x, y})
        });
        const js = await res.json();
        console.log("server:", js);
      } catch (err) {
        console.error(err);
      }
    });

    async function resetParking() {
      try {
        const res = await fetch('/reset', {method: 'POST'});
        const js = await res.json();
        console.log("reset:", js);
      } catch (err) {
        console.error(err);
      }
    }
  </script>
</body>
</html>
"""


# =========================
# 엔트리 포인트
# =========================
if __name__ == "__main__":
    # 백그라운드로 Detectron2 + 주차 로직 돌림
    t = threading.Thread(target=processing_loop, daemon=True)
    t.start()

    # Flask HTTP 서버 (GUI 필요 없음)
    app.run(host="0.0.0.0", port=8001, debug=True, threaded=True)
