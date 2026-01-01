# 1. 가상환경 생성 (ros2_ws 폴더 안에서)
python3 -m venv ~/yolo_venv --system-site-packages

# 2. 가상환경 활성화
source ~/yolo_venv/bin/activate

# 3. Ultralytics 설치
pip install ultralytics
--system-site-packages옵션을 쓰면 ROS 2가 설치한 기존 라이브러리들을 인식하면서도, YOLO에 필요한 추가 패키지만 안전하게 설치

# 가상환경이 활성화된 상태에서 아래 명령어로 집접 실행(개발/테스트)
python3 ~/ros2_ws/src/multi_sensor_fusion_pkg/multi_sensor_fusion_pkg/yolo_node.py