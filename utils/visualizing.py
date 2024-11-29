# from scipy.io import loadmat
# from pathlib import Path
# import numpy as np
# from matplotlib import pyplot as plt

# dataset_dir = Path('sitting_dataset/masks')
# dataset_masks = sorted(list(dataset_dir.glob('*.mat')))
# mat_data = loadmat(dataset_masks[0])

# print(np.unique(mat_data['M']))
# mask_visualize = np.zeros((mat_data['M'].shape[0], mat_data['M'].shape[1], 3))

# colors_bgr = [
#     (0, 0, 255),   # Red
#     (0, 255, 0),   # Green
#     (255, 0, 0),   # Blue
#     (0, 255, 255), # Yellow
#     (255, 255, 0), # Cyan
#     (255, 0, 255), # Magenta
#     (0, 128, 255), # Orange
#     (128, 255, 0), # Spring Green
#     (255, 128, 0), # Deep Pink
#     (0, 255, 128), # Sky Blue
#     (128, 0, 255), # Violet
#     (255, 128, 128), # Light Coral
#     (128, 255, 128), # Medium Spring Green
#     (128, 128, 255), # Light Steel Blue
#     (255, 255, 255)  # White
# ]
# for i in range(15):
#     mask_visualize[mat_data['M'] == i] = colors_bgr[i]
# mask_visualize = mask_visualize.astype(np.uint8)
# plt.imshow(mask_visualize)
# plt.legend()
# plt.show()

# from scipy.io import loadmat
# from pathlib import Path
# import numpy as np
# from matplotlib import pyplot as plt
# import matplotlib.patches as mpatches

# dataset_dir = Path('sitting_dataset/masks')
# dataset_masks = sorted(list(dataset_dir.glob('*.mat')))
# mat_data = loadmat(dataset_masks[0])
# print(np.unique(mat_data['M']))
# mask_visualize = np.zeros((mat_data['M'].shape[0], mat_data['M'].shape[1], 3))

# colors_bgr = [
#     (255, 0, 0),     # Red
#     (0, 255, 0),     # Green
#     (0, 0, 255),     # Blue
#     (255, 255, 0),   # Yellow
#     (255, 0, 255),   # Magenta
#     (0, 255, 255),   # Cyan
#     (128, 0, 0),     # Maroon
#     (0, 128, 0),     # Olive
#     (0, 0, 128),     # Navy
#     (128, 128, 0),   # Dark Yellow
#     (128, 0, 128),   # Purple
#     (0, 128, 128),   # Teal
#     (255, 165, 0),   # Orange
#     (165, 42, 42),   # Brown
#     (128, 128, 128)  # Gray
# ]

# import matplotlib.pyplot as plt

# # Create a figure
# fig, ax = plt.subplots()

# # Plot each color
# for i, color in enumerate(colors_bgr):
#     rect = plt.Rectangle((i, 0), 1, 1, color=np.array(color)/255)
#     ax.add_patch(rect)

# # Set axis limits and labels
# ax.set_xlim(0, len(colors_bgr))
# ax.set_ylim(0, 1)
# ax.set_xticks(range(len(colors_bgr)))
# ax.set_xticklabels([f'Color {i+1}' for i in range(len(colors_bgr))], rotation=45)
# ax.axis('off')

# plt.show()


# # Plot the mask
# for i in range(15):
#     mask_visualize[mat_data['M'] == i] = colors_bgr[i]

# mask_visualize = mask_visualize.astype(np.uint8)
# plt.imshow(mask_visualize)

# # Create a legend


# plt.show()


"""Class 0 - background
1 -head
2 - chest
3 - right forearm
4 - right arm
5 - right hand
6 - left forearm
7 - left arm
8 - left hand
9 - right lower leg
10 - right upper leg
11 - right foot
12 - left lower leg
13 - left upper leg
14 - left foot"""

