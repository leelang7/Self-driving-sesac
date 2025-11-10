import numpy as np
import cv2
from Line import Line


# =====================
# 기본 하이퍼파라미터
# =====================
# 허프 변환 및 엣지 탐지 튜닝 파라미터 (필요시 조절)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
GAUSSIAN_KERNEL_SIZE = 5

HOUGH_RHO = 2
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LEN = 20
HOUGH_MAX_LINE_GAP = 20

# 허용 기울기 범위 (노이즈 라인 제거)
MIN_ABS_SLOPE = 0.5    # 약 26도
MAX_ABS_SLOPE = 2.5    # 약 68도

# =====================
# 유틸 함수
# =====================

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Keeps only the region defined by vertices.
    """
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    vertices = np.array(vertices, dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask


def hough_lines_detection(img,
                          rho=HOUGH_RHO,
                          theta=HOUGH_THETA,
                          threshold=HOUGH_THRESHOLD,
                          min_line_len=HOUGH_MIN_LINE_LEN,
                          max_line_gap=HOUGH_MAX_LINE_GAP):
    """
    img: Canny 결과 (single channel)
    """
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    return lines


def weighted_img(overlay, base, alpha=0.8, beta=1.0, gamma=0.0):
    """
    alpha * base + beta * overlay + gamma
    """
    overlay = np.uint8(overlay)

    if len(overlay.shape) == 2:
        # grayscale → 3채널로 변환
        overlay = np.dstack((overlay, overlay, overlay))

    if overlay.shape[:2] != base.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]))

    return cv2.addWeighted(base, alpha, overlay, beta, gamma)


# =====================
# Lane 계산 로직
# =====================

def compute_lane_from_candidates(line_candidates, img_shape):
    """
    후보 Line 객체 리스트에서 좌/우 차선을 추정.
    """
    if not line_candidates:
        return None, None

    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    left_lane = None
    right_lane = None

    # 왼쪽 차선 (negative slope)
    if len(neg_lines) > 0:
        neg_biases = np.array([l.bias for l in neg_lines], dtype=np.float32)
        neg_slopes = np.array([l.slope for l in neg_lines], dtype=np.float32)

        neg_bias = np.median(neg_biases)
        neg_slope = np.median(neg_slopes)

        if neg_slope != 0:
            y1 = img_shape[0]
            y2 = int(img_shape[0] * 0.6)

            x1 = int((y1 - neg_bias) / neg_slope)
            x2 = int((y2 - neg_bias) / neg_slope)

            left_lane = Line(x1, y1, x2, y2)

    # 오른쪽 차선 (positive slope)
    if len(pos_lines) > 0:
        pos_biases = np.array([l.bias for l in pos_lines], dtype=np.float32)
        pos_slopes = np.array([l.slope for l in pos_lines], dtype=np.float32)

        pos_bias = np.median(pos_biases)
        pos_slope = np.median(pos_slopes)

        if pos_slope != 0:
            y1 = img_shape[0]
            y2 = int(img_shape[0] * 0.6)

            x1 = int((y1 - pos_bias) / pos_slope)
            x2 = int((y2 - pos_bias) / pos_slope)

            right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane


def get_lane_lines(color_image, solid_lines=True):
    """
    단일 프레임에서 차선 후보/대표 라인 계산.
    color_image: BGR 또는 RGB 프레임 (np.uint8)
    """
    # 내부 처리용 통일: 960x540로 리사이즈
    color_image = cv2.resize(color_image, (960, 540))

    # 그레이스케일
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # 블러 + 엣지
    img_blur = cv2.GaussianBlur(img_gray, (GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE), 0)
    img_edge = cv2.Canny(img_blur, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    # ROI 설정 (이미지 비율 기반으로 동적 계산)
    h, w = img_gray.shape
    roi_vertices = [[
        (int(0.05 * w), h),
        (int(0.45 * w), int(0.6 * h)),
        (int(0.55 * w), int(0.6 * h)),
        (int(0.95 * w), h),
    ]]
    img_edge_roi, _ = region_of_interest(img_edge, roi_vertices)

    # 허프 변환
    detected = hough_lines_detection(img_edge_roi)
    if detected is None:
        return [] if not solid_lines else (None, None)

    # (x1, y1, x2, y2) → Line 객체
    detected_lines = [Line(x1, y1, x2, y2) for [[x1, y1, x2, y2]] in detected]

    if not solid_lines:
        return detected_lines

    # 후보 필터링: 기울기 범위 내 라인만 사용
    candidate_lines = [
        l for l in detected_lines
        if MIN_ABS_SLOPE <= abs(l.slope) <= MAX_ABS_SLOPE
    ]

    if not candidate_lines:
        return (None, None)

    left_lane, right_lane = compute_lane_from_candidates(candidate_lines, img_gray.shape)
    return left_lane, right_lane


def smoothen_over_time(lane_lines_history):
    """
    여러 프레임에 대해 추정된 (left_lane, right_lane) 쌍 리스트를 받아
    평균 차선을 계산.
    lane_lines_history: [(left, right), (left, right), ...]
    """
    left_coords = []
    right_coords = []

    for lt, rt in lane_lines_history:
        if lt is not None:
            left_coords.append(lt.get_coords())
        if rt is not None:
            right_coords.append(rt.get_coords())

    avg_left = Line(*np.mean(left_coords, axis=0)) if left_coords else None
    avg_right = Line(*np.mean(right_coords, axis=0)) if right_coords else None

    return avg_left, avg_right


def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    """
    프레임 리스트를 받아 마지막 프레임 위에 차선을 그려 반환.
    frames: [frame1, frame2, ...] (각 frame: HxWx3)
            한 장이면 [frame] 형태로 넘기면 됨.
    """
    assert len(frames) >= 1, "frames 리스트가 비어 있습니다."

    is_videoclip = len(frames) > 1

    # 모든 프레임에 대해 차선 추정
    lane_lines_history = []
    for frame in frames:
        lanes = get_lane_lines(frame, solid_lines=solid_lines)
        lane_lines_history.append(lanes)

    # 시간적 스무딩
    if temporal_smoothing and solid_lines and is_videoclip:
        lane_lines = smoothen_over_time(lane_lines_history)
    else:
        lane_lines = lane_lines_history[-1]

    # 출력용 기준 프레임
    base_frame = cv2.resize(frames[-1], (960, 540))

    # 라인 그릴 마스크 (3채널)
    line_img = np.zeros_like(base_frame, dtype=np.uint8)

    # lane_lines가 (left, right) 튜플인 경우
    if solid_lines and isinstance(lane_lines, tuple):
        for lane in lane_lines:
            if lane is not None:
                lane.draw(line_img)

    # solid_lines=False 인 경우: 여러 Line 객체 리스트
    elif not solid_lines and isinstance(lane_lines, list):
        for lane in lane_lines:
            lane.draw(line_img)

    # ROI로 마스킹 (라인만 영역 제한)
    h, w = line_img.shape[:2]
    roi_vertices = [[
        (int(0.05 * w), h),
        (int(0.45 * w), int(0.6 * h)),
        (int(0.55 * w), int(0.6 * h)),
        (int(0.95 * w), h),
    ]]
    line_img_roi, _ = region_of_interest(line_img, roi_vertices)

    # 블렌딩
    blended = weighted_img(line_img_roi, base_frame, alpha=0.8, beta=1.0, gamma=0.0)
    return blended
