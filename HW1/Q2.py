import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter

def load_data(file_path):
    with open(file_path, "r") as file:
        data_lines = file.readlines()
    return data_lines[:700]

data_lines = load_data("./Q2")

def display_and_save_gif(direction='y', filename='animation.gif'):
    if direction == 'x':
        idx = 4
    elif direction == 'y':
        idx = 5
    elif direction == 'z':
        idx = 6
    else:
        raise ValueError("Invalid direction. Choose from 'x', 'y', or 'z'.")
    
    accel_data = data_lines[idx::7]
    accel_data = [float(val) for val in accel_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, len(accel_data))
    ax.set_ylim(min(accel_data) - 1, max(accel_data) + 1)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'Accelerometer Output ({direction}-direction)')
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = list(range(i))
        y = accel_data[:i]
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(fig, animate, init_func=init, frames=len(accel_data), interval=10, blit=True, repeat=False)
    
    writer = PillowWriter(fps=15)
    ani.save(filename, writer=writer)
    
    plt.tight_layout()
    plt.show()

    return filename

output_gif = display_and_save_gif(direction='z', filename='Q2_z_animation.gif')
