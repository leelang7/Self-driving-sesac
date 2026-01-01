import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torchvision.transforms as T
import cv2

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        self.sub = self.create_subscription(Image, '/front_camera/image_raw', self.callback, 10)
        self.bridge = CvBridge()
        self.model = torch.hub.load('pytorch/vision:v0.14.0', 'deeplabv3_resnet50', pretrained=True)
        self.model.eval()
        self.get_logger().info("Semantic Segmentation Node Started")

    def callback(self, msg):
        # 1. ROS 이미지 -> OpenCV (BGR)
        self.get_logger().info("이미지 수신 중...") # 이 로그가 터미널에 찍히는지 확인하세요!
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. 전처리 (BGR -> RGB 변환 및 정규화)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((480, 640)), # 크기를 고정하면 추론이 더 안정적입니다
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        inp = transform(rgb_frame).unsqueeze(0).to(torch.device('cpu')) # CPU 명시

        # 3. 추론
        with torch.no_grad():
            output = self.model(inp)['out'][0]
        
        # 4. 마스크 생성 (가장 높은 확률의 클래스 인덱스 추출)
        mask = output.argmax(0).cpu().numpy().astype('uint8')

        # 5. 시각화 (까만 화면 해결 포인트!)
        # 클래스 인덱스(0, 1, 2...)는 너무 작아서 그냥 보면 까맣게 보입니다.
        # 클래스 간 구분을 위해 값을 12배 정도 키우고(약 20개 클래스 대응), 
        # 0(배경)이 아닌 값들이 확실히 보이게 합니다.
        mask_visual = (mask * 12).astype('uint8') 
        colored_mask = cv2.applyColorMap(mask_visual, cv2.COLORMAP_JET)

        # 배경(index 0)은 검은색으로 강제 지정
        colored_mask[mask == 0] = [0, 0, 0]

        # 6. 결과 출력
        # 원본 이미지와 마스크를 나란히 붙여서 확인하면 더 정확합니다.
        resized_frame = cv2.resize(frame, (640, 480))
        combined = cv2.hconcat([resized_frame, colored_mask])
        
        cv2.imshow("Semantic Segmentation (Left: Raw, Right: Mask)", combined)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
