import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torchvision.transforms as T
import cv2
import time # 시간 측정을 위해 추가

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        # 1. GPU 장치 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        self.sub = self.create_subscription(Image, '/front_camera/image_raw', self.callback, 10)
        self.bridge = CvBridge()
        
        # 2. 모델 로드 및 GPU로 이동
        self.model = torch.hub.load('pytorch/vision:v0.14.0', 'deeplabv3_resnet50', weights='DeepLabV3_ResNet50_Weights.DEFAULT')
        self.model.to(self.device)
        self.model.eval()
        
        # FPS 계산용 변수
        self.prev_time = 0
        self.fps = 0
        
        self.get_logger().info("Semantic Segmentation Node Started with GPU & FPS counter")

    def callback(self, msg):
        # 시간 측정 시작
        start_time = time.time()
        
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((480, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        inp = transform(rgb_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(inp)['out'][0]
        
        mask = output.argmax(0).cpu().numpy().astype('uint8')

        # 시각화 로직
        mask_visual = (mask * 12).astype('uint8') 
        colored_mask = cv2.applyColorMap(mask_visual, cv2.COLORMAP_JET)
        colored_mask[mask == 0] = [0, 0, 0]

        resized_frame = cv2.resize(frame, (640, 480))
        combined = cv2.hconcat([resized_frame, colored_mask])

        # --- FPS 계산 및 표시부 ---
        end_time = time.time()
        curr_fps = 1.0 / (end_time - start_time)
        # 지수 이동 평균으로 FPS 부드럽게 (새로운 값 10% 반영)
        self.fps = (self.fps * 0.9) + (curr_fps * 0.1)
        
        fps_text = f"FPS: {self.fps:.1f} ({self.device})"
        # 좌상단에 텍스트 넣기 (검은색 테두리에 흰색 글씨로 가독성 확보)
        cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        # -------------------------
        
        cv2.imshow("Semantic Segmentation", combined)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()