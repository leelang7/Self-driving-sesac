import numpy as np
import cv2


class Line:
    """
    A geometric line defined by two endpoints (x1, y1) and (x2, y2).
    Supports slope, bias computation, coordinate updates, and drawing.
    """

    def __init__(self, x1, y1, x2, y2):
        # 모든 좌표는 float32로 저장하되, 연산 안정성을 위해 numpy 사용
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def compute_slope(self):
        """기울기(slope)를 계산 (분모=0 방지용 epsilon 포함)."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return dy / (dx + np.finfo(np.float32).eps)

    def compute_bias(self):
        """y절편(bias) 계산."""
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        """(x1, y1, x2, y2) 반환."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def set_coords(self, x1, y1, x2, y2):
        """좌표 갱신 및 slope/bias 재계산."""
        self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def draw(self, img, color=(255, 0, 0), thickness=8, antialias=True):
        """
        이미지 위에 라인을 그림.
        - color: (B, G, R)
        - thickness: 두께 (픽셀)
        - antialias=True 시 cv2.LINE_AA로 부드럽게 렌더링
        """
        if img is None or img.size == 0:
            return

        p1 = (int(round(self.x1)), int(round(self.y1)))
        p2 = (int(round(self.x2)), int(round(self.y2)))

        line_type = cv2.LINE_AA if antialias else cv2.LINE_8
        cv2.line(img, p1, p2, color, thickness, line_type)

    def length(self):
        """라인 길이 반환."""
        return np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    def angle(self, degrees=True):
        """라인의 각도 반환 (default: degree)."""
        angle = np.arctan2(self.y2 - self.y1, self.x2 - self.x1)
        return np.degrees(angle) if degrees else angle

    def __repr__(self):
        return f"Line(({self.x1:.1f}, {self.y1:.1f}) → ({self.x2:.1f}, {self.y2:.1f}), slope={self.slope:.3f})"
