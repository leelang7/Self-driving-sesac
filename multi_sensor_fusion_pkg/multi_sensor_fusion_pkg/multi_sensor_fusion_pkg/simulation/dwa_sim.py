import numpy as np
import matplotlib.pyplot as plt

# 로봇 파라미터
dt = 0.1
v_max, w_max = 1.0, 1.0
obstacles = np.array([[2.0, 2.0], [2.5, 1.5], [3.0, 3.0]])
goal = np.array([5.0, 5.0])

def motion(x, u):
    x[0] += u[0] * np.cos(x[2]) * dt
    x[1] += u[0] * np.sin(x[2]) * dt
    x[2] += u[1] * dt
    return x

def calc_score(x, u):
    # heading
    heading = np.dot(goal - x[:2], [np.cos(x[2]), np.sin(x[2])])
    # distance
    dists = np.linalg.norm(obstacles - x[:2], axis=1)
    min_dist = np.min(dists)
    # velocity
    vel_score = u[0]
    return 0.8*heading + 0.2*min_dist + 0.5*vel_score

x = np.array([0.0, 0.0, 0.0])
path = [x[:2].copy()]

for t in np.arange(0, 10, dt):
    best_score = -1e9
    best_u = [0.0, 0.0]
    for v in np.linspace(0, v_max, 5):
        for w in np.linspace(-w_max, w_max, 5):
            x_tmp = x.copy()
            x_tmp = motion(x_tmp, [v, w])
            score = calc_score(x_tmp, [v, w])
            if score > best_score:
                best_score, best_u = score, [v, w]
    x = motion(x, best_u)
    path.append(x[:2].copy())

path = np.array(path)
plt.figure(figsize=(6,6))
plt.plot(path[:,0], path[:,1], 'b-', label="Path")
plt.plot(goal[0], goal[1], 'go', label="Goal")
plt.scatter(obstacles[:,0], obstacles[:,1], c='r', label="Obstacles")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Dynamic Window Approach Simulation")
plt.show()
