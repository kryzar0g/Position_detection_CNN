import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.animation as animation

POSE_CONNECTIONS = [
    (28, 30), (30, 32), (28, 32), (26, 28), (24, 26),
    (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (11, 23), (11, 12), (12, 24), (12, 14),
    (14, 16), (16, 18), (18, 20), (16, 20), (16, 22),
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 19),
    (15, 21), (8, 6), (6, 5), (5, 4), (4, 0), (0, 1),
    (1, 2), (2, 3), (3, 7), (9, 10)
]

# Read CSV data
with open('pose_landmarks.csv', mode='r') as file:
    joint_pos = csv.reader(file)
    rows = list(joint_pos)

# Extract headers
headers = rows[0]

# Prepare figure
fig, ax = plt.subplots()
line, = ax.plot([], [], 'ro')


# Animation function
def update(frame):
    ax.clear()
    xpoints, ypoints = [], []

    if len(rows) > frame:
        num_cols = len(rows[frame])
        # Extract X and Y points
        for col in range(1, num_cols, 4):  # X coordinates
            xpoints.append(float(rows[frame][col]))
        for col in range(2, num_cols, 4):  # Y coordinates
            ypoints.append(float(rows[frame][col]))

        # Plot points
        ax.scatter(xpoints, ypoints, color='blue')

        # Plot connections
        for connection in POSE_CONNECTIONS:
            start, end = connection
            if start < len(xpoints) and end < len(xpoints):
                ax.plot(
                    [xpoints[start], xpoints[end]],
                    [ypoints[start], ypoints[end]],
                    color="red"
                )

        # Annotate points
        for i, txt in enumerate(headers[1:num_cols:4]):
            ax.annotate(txt, (xpoints[i], ypoints[i]))

    return line,

# Set up animation
ani = animation.FuncAnimation(fig, update, frames=range(1,1000), blit=False, repeat=False)

# Save as GIF
ani.save("pose_landmarks.gif", writer='pillow', fps=60)

plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.title("Pose Landmarks")
plt.grid()
plt.gca().invert_yaxis()
plt.show()
