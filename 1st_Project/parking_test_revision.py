import cv2
import numpy as np
import math
from dataclasses import dataclass

# =========================
# 설정
# =========================
USE_DUMMY_SLOTS = False  # True면 YOLO 대신 더미 슬롯 사용
VIDEO_SOURCE = "parking_test_video.mp4"  # 0이면 웹캠

FRAME_W = 1280
FRAME_H = 720

YOLO_WEIGHTS = "class29_seg.pt"
PARKING_CLASS_ID = 4  # Parking Area 클래스 ID (모델에 맞게 조정)


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


# =========================
# YOLO ParkingSlot Detector
# =========================
class ParkingDetector:
    def __init__(self, use_dummy=True):
        self.use_dummy = use_dummy
        self.slot_id_counter = 0
        if not use_dummy:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_WEIGHTS)
        else:
            self.model = None

    def detect(self, frame):
        if self.use_dummy:
            return self._dummy_slots(frame)
        return self._yolo_slots(frame)

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

    def _yolo_slots(self, frame):
        """YOLO-seg 결과에서 Parking Area만 가져옴."""
        results = self.model(frame, verbose=False)[0]

        if results.masks is None:
            return []

        polys = results.masks.xy
        classes = results.boxes.cls.cpu().numpy().astype(int)

        slots = []
        self.slot_id_counter = 0

        for poly, cid in zip(polys, classes):
            if cid != PARKING_CLASS_ID:
                continue

            poly = poly.astype(np.int32)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            self.slot_id_counter += 1
            slots.append(ParkingSlot(self.slot_id_counter, poly, (cx, cy)))

        return slots


# =========================
# Slot Selector (마우스)
# =========================
class SlotSelector:
    def __init__(self):
        self.slots = []
        self.selected_slot_id = None

    def update(self, slots):
        self.slots = slots

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
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
# Parking Planner (전진 → 후진, 단순 보간)
# =========================
class ParkingPlanner:
    def __init__(self):
        self.path = []

    def plan(self, start_pose: CarPose, goal_center: tuple):
        """
        1) start_pose  → 접근 위치(전진, 차선 방향 그대로)
        2) 접근 위치   → goal_center(후진, yaw -pi/2 → 0으로 회전)
        """
        gx, gy = goal_center

        # 최종 주차 방향: 가로로 눕게 (오른쪽을 보는 방향)
        slot_yaw = 0.0

        park_pose = CarPose(gx, gy, slot_yaw, reverse=True)

        # 접근 위치: 슬롯 아래쪽, yaw는 차선 방향(-pi/2) 유지
        approach_dist = 180
        ax = gx
        ay = gy + approach_dist
        approach_pose = CarPose(ax, ay, -math.pi / 2, reverse=False)

        forward = self._interp_path(start_pose, approach_pose,
                                    reverse=False, smooth=False)
        reverse = self._interp_path(approach_pose, park_pose,
                                    reverse=True, smooth=True)

        self.path = forward + reverse
        print(f"[INFO] 경로 생성 (전진 {len(forward)}점, 후진 {len(reverse)}점)")
        return self.path

    def _interp_path(self, p0: CarPose, p1: CarPose,
                     reverse: bool, smooth: bool):
        steps = 50
        path = []
        for i in range(steps):
            t = i / (steps - 1)
            x = p0.x * (1 - t) + p1.x * t
            y = p0.y * (1 - t) + p1.y * t
            if smooth:
                yaw = self._smooth_yaw(p0.yaw, p1.yaw, t)
            else:
                yaw = p0.yaw * (1 - t) + p1.yaw * t
            path.append(CarPose(x, y, yaw, reverse=reverse))
        return path

    @staticmethod
    def _smooth_yaw(y0, y1, t):
        # S-curve 보간
        t2 = t * t * (3 - 2 * t)
        return y0 * (1 - t2) + y1 * t2


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

    # 전진 / 후진에 따라 "앞" 표시 위치 변경
    if not p.reverse:
        front = rect[1]   # 전진 때 앞
    else:
        front = rect[3]   # 후진 때는 반대쪽을 앞처럼 보이게
    cv2.circle(img, tuple(front), 6, (0, 0, 255), -1)


# =========================
# MAIN
# =========================
def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    detector = ParkingDetector(USE_DUMMY_SLOTS)
    selector = SlotSelector()
    planner = ParkingPlanner()
    follower = PathFollower()
    bev_renderer = BEVRenderer()

    cv2.namedWindow("FrontView")
    cv2.setMouseCallback("FrontView", selector.on_mouse)

    # 시작 차량 위치: 아래 중앙, 위쪽을 바라봄
    car_pose = CarPose(FRAME_W // 2, int(FRAME_H * 0.9), -math.pi / 2)

    freeze = False
    freeze_frame = None
    freeze_slots = []

    slots = []
    path = []

    while True:
        # ----- 영상/YOLO -----
        if not freeze:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            slots = detector.detect(frame)
            selector.update(slots)
            current = frame.copy()
            slots_to_draw = slots
            mode_text = "SEARCH (click slot to park, 'c' to reset)"
        else:
            current = freeze_frame.copy()
            slots_to_draw = freeze_slots
            mode_text = "PARKING (press 'c' to cancel)"

        selected = selector.get()

        # ----- 슬롯 선택 → Freeze + 경로 생성 -----
        if selected is not None and not freeze:
            print("[INFO] Freeze ON & Path Planning")
            freeze = True
            freeze_frame = current.copy()
            freeze_slots = slots.copy()

            path = planner.plan(
                CarPose(car_pose.x, car_pose.y, car_pose.yaw),
                selected.center
            )
            follower.start(path)

        # ----- 경로 추종 -----
        if follower.active:
            new_pose = follower.step()
            if new_pose is not None:
                car_pose = new_pose

        # ----- FrontView 렌더링 -----
        overlay = current.copy()

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

        # ----- BEV 렌더링 -----
        bev = bev_renderer.render(slots_to_draw, car_pose, path)

        cv2.imshow("FrontView", overlay)
        cv2.imshow("BEV", bev)

        # ----- 키 입력 -----
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('c'):
            print("[INFO] Reset / Cancel parking")
            freeze = False
            selector.clear()
            follower.active = False
            path = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
