import matplotlib.pyplot as plt
import numpy as np

dt = 0.1
time = np.arange(0, 10, dt)

# PID 계수
Kp, Ki, Kd = 1.0, 0.1, 0.05

# 목표 속도
target = 1.0

# 초기값
v = 0.0
integral = 0.0
prev_error = 0.0
v_list = []

for t in time:
    error = target - v
    integral += error * dt
    derivative = (error - prev_error) / dt
    u = Kp * error + Ki * integral + Kd * derivative
    v += u * dt  # 속도 변화
    prev_error = error
    v_list.append(v)

plt.plot(time, v_list, label='Speed')
plt.axhline(y=target, color='r', linestyle='--', label='Target')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('PID Speed Control Simulation')
plt.legend()
plt.grid(True)
plt.show()
