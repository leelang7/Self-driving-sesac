## <자율주행 프로젝트 환경 구축 및 데모>



## 🚀 올바른 설치 & 실행 절차 (Ubuntu 22.04 + NVIDIA GPU)

- 자율주행 개발 환경 구축의 난이도는 상급으로 cuda, python, opencv, numpy, tensorflow, pytorch, ros2, 시뮬레이터 등 여러 환경이 복합적으로 구성되어 있으므로 도커 사용을 적극 권장한다. 직접 구축을 한다면 많은 시간적 리소스 소비와 실패할 확률이 매우 높음을 사전에 알린다. 도커의 위대함을 체감하고 싶다면 직접 실패 경험을 해보는 것도 나쁘지 않다.
- sudo apt update시 기존 gazebo classic 과 호환되는 라이브러리들을 더이상 제공하지 않기 때문에(Gazebo fortess/Garden 사용 장려 등 이슈로 인해) 직접 구축은 사실상 낭비에 가깝다.
- Ubuntu 22.04에서 구축하며 실제 컨테이너 내부는 20.04로 되어 있다. 이는 Gazebo 업데이트와 3d 모델링 파일들의 구조들을 다른 환경으로 포팅하는 것이 매우 소모적이기에 개발/테스트가 완료된 시점에 동기화 되어 있다.(Gazebo classic 조합)

### ✅ 1️⃣ Docker + NVIDIA runtime 설치

```
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Docker GPG 등록
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Docker repo 추가
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
newgrp docker
```

------

### ✅ 2️⃣ NVIDIA runtime 세팅

```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

> 테스트:
>
> ```
> docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
> ```
>
> → GPU 목록 뜨면 OK.

------

### ✅ 3️⃣ Docker 이미지 가져오기 (pull)

```
docker pull noshluk2/ros2-self-driving-car-ai-using-opencv:latest
```

> 이건 `ros2_self_driving_car`를 직접 빌드할 필요가 없음.
>  Docker Hub에 이미 완성된 환경이 올라가 있음.

------

### ✅ 4️⃣ 컨테이너 생성 (bash 파일 이용)

```
cd ~/sesac_ws/src/ROS2-Self-Driving-Car-AI-using-OpenCV/docker
chmod +x create_container.bash
./create_container.bash
```

> ⚠️ 이건 한 번만 실행해야 함.
>  계속 실행하면 `ros2_sdc_container`가 여러 개 생김.
>
> **./create_container.bash를 하면 생성과 동시에 도커 컨테이너로 진입되니 참고(프롬프트 기억)**

------

### ✅ 5️⃣ 컨테이너 재진입 / 터미널 연결

```
docker start ros2_sdc_container
docker exec -it ros2_sdc_container bash
```

------

### ✅ 6️⃣ 시뮬레이터 실행

```
ros2 launch self_driving_car_pkg world_gazebo.launch.py
```

그리고 새 터미널 하나 더 연결(새로운 컨테이너)해서:

```
docker exec -it ros2_sdc_container bash
cd ~/ROS2-Self-Driving-Car-AI-using-OpenCV/
ros2 run self_driving_car_pkg computer_vision_node
```

------

## ✅ 핵심 요약

| 단계 | 명령                                                         | 설명              |
| ---- | ------------------------------------------------------------ | ----------------- |
| 1    | Docker 설치                                                  | Ubuntu 22.04 기준 |
| 2    | nvidia-docker2 설치                                          | GPU 가속          |
| 3    | `docker pull noshluk2/ros2-self-driving-car-ai-using-opencv` | 도커 이미지 pull  |
| 4    | `./create_container.bash`                                    | 컨테이너 1개 생성 |
| 5    | `docker exec -it ros2_sdc_container bash`                    | 컨테이너 진입     |
| 6    | `ros2 launch self_driving_car_pkg world_gazebo.launch.py`    | Gazebo 실행       |
| 7    | `ros2 run self_driving_car_pkg computer_vision_node`         | AI 주행 노드 실행 |

------





### <도커 관리 및 개발/테스트/실험 운영>

- 호스트와 동기화하면 좋지만 본 프로젝트 특성상 cuda, gui 이슈 등의 조건으로 또 다른 문제를 야기시킬 수 있어 번거롭지만 commit 활용

  

### ※ 도커 Commit(컨테이너 내부 코드 수정 등) - 중요

- 컨테이터 외부 일반 command에서 실행해야함(개념 중요)

```
docker commit ros2_sdc_container ros2_sdc_fixed:latest
```

- 다른 이미지 파일로 백업되니 만약 차후에 아예 깨졌더라고 환경 그대로 복원 사용 가능

  

### 1️⃣ 현재 상태 확인

```
docker ps -a
```

- STATUS가 `Exited`면 그냥 꺼진 것
   → 재시작 가능:

  ```
  docker start -ai ros2_sdc_container
  ```

- 만약 리스트에도 아예 없으면 (삭제된 경우): 다음 단계로.

------

###  2️⃣ 백업 이미지로 새 컨테이너 다시 생성

`ros2_sdc_fixed` 이미지를 이용해 새 컨테이너 만들면 완벽 복원돼.

```
docker run -it --net=host --gpus all \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --name ros2_sdc_container \
  --privileged --runtime=nvidia \
  ros2_sdc_fixed:latest \
  bash
```

👉 이렇게 하면:

- 예전 컨테이너에서 수정한 코드, 모델, 세팅 그대로 살아남음
- 단, 커밋 시점 이후의 변경사항은 반영되지 않음

------

### 3️⃣ 재시작만으로 복구되는 경우 (자주 쓰는 방식)

컨테이너가 “삭제”된 게 아니라 “꺼진” 거면
 다시 켜는 것만으로 그대로 이어짐:

```
docker start -ai ros2_sdc_container
```

이건 진짜 “그 자리에서 pause → resume” 하는 거라
 **파일, ROS 세션, 환경 그대로 살아 있음**
