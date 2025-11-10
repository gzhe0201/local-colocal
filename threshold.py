import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load your image (replace with your file path)
img = cv2.imread(r"C:\Users\m319725\Desktop\Test python\H1299 images adjusted for presentation\H1299_0.7Gy_1hr_c140_gH2AX_53bp1_3_adjusted.tif", cv2.IMREAD_UNCHANGED)
  # keep original depth
b, g, r = cv2.split(img)  # choose the channel you want
channel = b  # e.g., DAPI channel

# Initial threshold value
init_thresh = 50

# Create figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
mask = np.zeros_like(channel)
im = ax.imshow(mask, cmap='gray')
ax.set_title(f"Threshold = {init_thresh}")

# Slider axis
ax_thresh = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_thresh, 'Threshold', 0, 255, valinit=init_thresh, valstep=1)

# Update function
def update(val):
    thresh = slider.val
    _, mask = cv2.threshold(channel, thresh, 255, cv2.THRESH_BINARY)
    im.set_data(mask)
    ax.set_title(f"Threshold = {int(thresh)}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
