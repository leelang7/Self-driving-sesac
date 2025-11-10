import os
import sys
import time
from os.path import join, basename, dirname, abspath
from collections import deque

import cv2
from lane_detection import color_frame_pipeline


# =========================
# 설정
# =========================
BASE_DIR = dirname(abspath(__file__))
DATA_DIR = join(BASE_DIR, 'data')

RESIZE_W, RESIZE_H = 960, 540
TEMPORAL_WINDOW = 10            # 스무딩용 프레임 개수
ENABLE_DISPLAY = True           # GUI 환경 아니면 False로
OUTPUT_ENABLED = True           # 결과 영상 파일 저장 여부

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def pick_default_source():
    """인자 없을 때 기본 입력 선택: test_videos에 있으면 그거, 없으면 웹캠 0."""
    test_videos_dir = join(DATA_DIR, 'test_videos')
    if os.path.isdir(test_videos_dir):
        files = [
            join(test_videos_dir, f)
            for f in sorted(os.listdir(test_videos_dir))
            if f.lower().endswith(VIDEO_EXTS)
        ]
        if files:
            print(f"[INFO] 기본 비디오 소스 사용: {files[0]}")
            return files[0]
    print("[INFO] test_videos 없음 또는 비어있음 → 웹캠(0) 사용 시도")
    return 0  # webcam


def parse_source_arg():
    """CLI 인자 해석: 숫자면 카메라 인덱스, 아니면 파일 경로."""
    if len(sys.argv) < 2:
        return pick_default_source()

    src = sys.argv[1]

    # 숫자면 웹캠 인덱스로 처리
    if src.isdigit():
        return int(src)

    # 경로면 그대로
    return src


def open_capture(source):
    """source에 따라 VideoCapture 생성."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERR] 입력 소스 열기 실패: {source}")
        return None
    return cap


def get_source_fps(cap, is_camera: bool):
    """파일이면 메타데이터 fps 사용, 카메라면 cap에서 읽되 실패시 기본값."""
    fps = cap.get(cv2.CAP_PROP_FPS)

    if is_camera:
        # 웹캠은 드라이버마다 0 나오기도 해서, 이상하면 30 가정
        if fps is None or fps <= 0 or fps != fps:
            fps = 30.0
    else:
        # 파일인데 fps 정보 없으면 30 가정
        if fps is None or fps <= 0 or fps != fps:
            fps = 30.0

    return fps


def create_writer(out_dir, source_name, fps):
    """출력 VideoWriter 생성 (선택)."""
    ensure_dir(out_dir)
    out_path = join(out_dir, f"lane_{basename(str(source_name))}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (RESIZE_W, RESIZE_H))
    if not writer.isOpened():
        print(f"[WARN] VideoWriter 생성 실패: {out_path}")
        return None, None
    return writer, out_path


def run():
    source = parse_source_arg()
    is_camera = isinstance(source, int)

    cap = open_capture(source)
    if cap is None:
        return

    fps_in = get_source_fps(cap, is_camera)
    print(f"[INFO] 입력 소스: {source} | 타입: {'CAM' if is_camera else 'VIDEO'} | FPS: {fps_in:.2f}")

    out_writer = None
    out_path = None
    if OUTPUT_ENABLED:
        out_dir = join(BASE_DIR, "out", "videos")
        out_writer, out_path = create_writer(out_dir, source, fps_in)

    frame_buffer = deque(maxlen=TEMPORAL_WINDOW)

    prev_time = time.time()
    smoothed_fps = 0.0
    alpha = 0.9  # 표시용 FPS EMA

    while True:
        start = time.time()
        ret, frame_bgr = cap.read()
        if not ret:
            if is_camera:
                # 카메라면 잠깐 끊겼을 수도 있으니 계속 시도 가능하지만,
                # 여기선 단순 종료 처리
                print("[INFO] 카메라 입력 종료.")
            else:
                print("[INFO] 비디오 끝.")
            break

        # 리사이즈 & RGB 변환
        frame_bgr = cv2.resize(frame_bgr, (RESIZE_W, RESIZE_H))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_buffer.append(frame_rgb)

        # 차선 검출 + 시간 스무딩
        blend_rgb = color_frame_pipeline(
            frames=list(frame_buffer),
            solid_lines=True,
            temporal_smoothing=True
        )
        blend_bgr = cv2.cvtColor(blend_rgb, cv2.COLOR_RGB2BGR)

        # 실제 루프 FPS 계산 (재생 속도 기반)
        now = time.time()
        inst_fps = 1.0 / (now - prev_time + 1e-9)
        smoothed_fps = alpha * smoothed_fps + (1 - alpha) * inst_fps
        prev_time = now

        # FPS & 타입 표시
        label = f"{'CAM' if is_camera else 'VID'} FPS: {smoothed_fps:5.1f}"
        cv2.putText(
            blend_bgr,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # 출력 저장
        if out_writer is not None:
            out_writer.write(blend_bgr)

        # 화면 표시 + 재생속도 동기화
        if ENABLE_DISPLAY:
            # 처리 시간 포함해서 원본 fps에 맞게 딜레이 계산
            proc_time = time.time() - start
            ideal_frame_time = 1.0 / fps_in
            wait_sec = max(ideal_frame_time - proc_time, 0)
            wait_ms = max(int(wait_sec * 1000), 1)

            cv2.imshow("Lane Detection (ESC to quit)", blend_bgr)
            if cv2.waitKey(wait_ms) & 0xFF == 27:  # ESC
                print("[INFO] ESC 입력으로 종료.")
                break

    cap.release()
    if out_writer is not None:
        out_writer.release()
        print(f"[INFO] 결과 저장: {out_path}")

    if ENABLE_DISPLAY:
        cv2.destroyAllWindows()

    print("[DONE] 종료.")


if __name__ == "__main__":
    run()
