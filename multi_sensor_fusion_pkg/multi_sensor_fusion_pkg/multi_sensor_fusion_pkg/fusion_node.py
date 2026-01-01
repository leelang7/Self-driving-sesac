import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # 구독자 설정 (네임스페이스 확인 필요)
        self.create_subscription(Image, '/front/image_raw', self.image_callback, 10)
        self.create_subscription(LaserScan, '/lidar/lidar_scan', self.lidar_callback, 10)
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        self.bridge = CvBridge()
        self.latest_imu = None

        # 시각화 설정 (깜빡임 방지 및 각도 보정)
        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.ax.set_theta_zero_location("N")  # 정면(0도)을 위쪽(북쪽)으로 설정
        self.ax.set_theta_direction(-1)       # 시계 방향으로 각도 증가
        self.ax.set_ylim(0, 10)
        
        # 업데이트할 데이터 객체 미리 생성
        self.lidar_line, = self.ax.plot([], [], 'b.', markersize=2)
        self.imu_pointer, = self.ax.plot([], [], 'r-', linewidth=2, label='IMU Heading')
        
        self.get_logger().info("FusionNode started. Graphs initialized.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Front Camera", cv2.resize(frame, (320, 240)))
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Image Error: {e}")

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        # np.arange 대신 np.linspace를 써서 개수 불일치(ValueError) 해결
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # 데이터 정제 (inf, nan 처리)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 0.0, ranges)
        ranges = np.clip(ranges, 0, 10)

        # 데이터만 업데이트 (ax.clear() 안 함 -> 깜빡임 해결)
        self.lidar_line.set_data(angles, ranges)

        if self.latest_imu is not None:
            yaw = self.quaternion_to_yaw(self.latest_imu.orientation)
            # IMU 방향을 가리키는 빨간 선 업데이트
            self.imu_pointer.set_data([yaw, yaw], [0, 10])
            self.ax.set_title(f"LiDAR + IMU (Yaw: {math.degrees(yaw):.1f}°)", pad=20)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def imu_callback(self, msg):
        self.latest_imu = msg

    def quaternion_to_yaw(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        plt.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()