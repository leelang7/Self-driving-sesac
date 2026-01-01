import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import torch
import time

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()
        
        # 1. GPU 장치 설정 및 모델 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt')
        self.model.to(self.device) # 모델을 GPU로 이동
        
        self.sub = self.create_subscription(Image, '/front_camera/image_raw', self.callback, 10)
        
        # FPS 계산용 변수
        self.prev_time = 0
        self.fps = 0
        
        self.get_logger().info(f"YOLOv8 ROS2 Detector Started on {self.device}")

    def callback(self, msg):
        # 시간 측정 시작
        start_time = time.time()
        
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. GPU 추론 실행 (device 파라미터 명시)
        # verbose=False를 주면 터미널이 깔끔해집니다.
        results = self.model.predict(frame, device=self.device, verbose=False)
        
        # 결과 렌더링
        annotated = results[0].plot()
        
        # --- FPS 계산 및 표시 ---
        end_time = time.time()
        curr_fps = 1.0 / (end_time - start_time)
        self.fps = (self.fps * 0.9) + (curr_fps * 0.1) # EMA 필터
        
        fps_text = f"FPS: {self.fps:.1f} ({self.device})"
        # 좌상단 텍스트 (가독성을 위한 외곽선 포함)
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        # -----------------------

        cv2.imshow("YOLO Detection", annotated)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()