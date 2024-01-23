import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def polar_to_cartesian(r, theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    x = r * np.cos(theta_rad)
    y = r * np.sin(theta_rad)
    return x, y

data = np.genfromtxt('./Q1.dat')

distances = data[:, 0]
angles = data[:, 1]
timestamps = data[:, 2]

x_coords, y_coords = polar_to_cartesian(distances, angles)

unique_timestamps = np.unique(timestamps)
grouped_data = {timestamp: [] for timestamp in unique_timestamps}
for x, y, timestamp in zip(x_coords, y_coords, timestamps):
    grouped_data[timestamp].append((x, y))

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.grid(True)

line, = ax.plot([], [], 'o', markersize=2)

def init():
    line.set_data([], [])
    return line,

def animate(timestamp):
    points = grouped_data[timestamp]
    xs, ys = zip(*points)
    line.set_data(xs, ys)
    ax.set_title(f"Timestamp: {timestamp:.3f}")
    return line,

ani = FuncAnimation(fig, animate, frames=unique_timestamps, init_func=init, blit=True, repeat=False)

writer = PillowWriter(fps=7)
ani.save("Q1.gif", writer=writer)

plt.show()
