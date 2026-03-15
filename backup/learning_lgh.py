import matplotlib.pyplot as plt
import numpy as np

# 1. Setup the Environment
obs_pos = np.array([0.0, 0.0])
obs_radius = 1.5
robot_pos = np.array([2.5, 2.0])

# 2. Calculate the CBF Math
x_diff = robot_pos - obs_pos
dist = np.linalg.norm(x_diff)
h_x = dist - obs_radius

# L_g h: The normalized unit vector pointing directly away from the obstacle
L_g_h = x_diff / dist

# Proposed input velocity u (Robot wants to drive aggressively up and to the left)
u = np.array([-2.0, 0.5])

# The Dot Product: How much of 'u' is pointing in the direction of 'L_g h'
dot_product = np.dot(L_g_h, u)

# The "Shadow" (Projection): Visualizing the dot product as a vector
projection = dot_product * L_g_h

# 3. Plotting Setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)

# Draw Obstacle
circle = plt.Circle(obs_pos, obs_radius, color='red', alpha=0.3, label='Obstacle')
ax.add_patch(circle)

# Draw Robot
ax.plot(robot_pos[0], robot_pos[1], 'bo', markersize=8, label='Robot')

# Draw L_g h (The Outward Gradient / Laser Beam)
# Scaled up slightly for visual clarity
ax.quiver(robot_pos[0], robot_pos[1], L_g_h[0]*1.5, L_g_h[1]*1.5, 
          angles='xy', scale_units='xy', scale=1, color='green', 
          width=0.008, label='$L_g h$ (Gradient)')

# Draw u (The Requested Velocity)
ax.quiver(robot_pos[0], robot_pos[1], u[0], u[1], 
          angles='xy', scale_units='xy', scale=1, color='orange', 
          width=0.008, label='$u$ (Requested Velocity)')

# Draw the Projection (The "Shadow" on the laser beam)
ax.quiver(robot_pos[0], robot_pos[1], projection[0], projection[1], 
          angles='xy', scale_units='xy', scale=1, color='purple', 
          width=0.008, label='Projection ($L_g h \cdot u$)')

# Draw dashed lines to show how the shadow is cast perfectly perpendicular
ax.plot([robot_pos[0] + u[0], robot_pos[0] + projection[0]], 
        [robot_pos[1] + u[1], robot_pos[1] + projection[1]], 
        'k--', alpha=0.5)

# Formatting
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_title(f"CBF Radar Gun\nDot Product = {dot_product:.2f} m/s", fontsize=14)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.legend(loc='lower left')

plt.show()