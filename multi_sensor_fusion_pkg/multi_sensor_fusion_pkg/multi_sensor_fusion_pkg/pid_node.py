import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class PIDController(Node):
    def __init__(self):
        super().__init__('pid_controller')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.target_x = 2.0  # 목표 위치
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05
        self.integral = 0.0
        self.prev_error = 0.0
        self.current_x = 0.0
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("PID Controller node started.")

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x

    def control_loop(self):
        error = self.target_x - self.current_x
        self.integral += error * 0.1
        derivative = (error - self.prev_error) / 0.1
        control = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        twist = Twist()
        twist.linear.x = max(min(control, 0.3), -0.3)  # 속도 제한
        self.publisher_.publish(twist)

        self.get_logger().info(f"Target={self.target_x:.2f}, Pos={self.current_x:.2f}, Cmd={twist.linear.x:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = PIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
