import numpy as np
data = np.load('models/ssm_model.npz')
print('SSM keys:', list(data.keys()))
print('grid_shape:', data['grid_shape'])
print('components:', data['components'].shape)
r = data['explained_variance_ratio']
print('cumsum first 20:', np.cumsum(r)[:20].round(3))
print('train_subjects count:', len(data['train_subjects']))

# Compute approximate voxel spacings
# The occupancy bounds from prepare_occupancy_targets.py:
# X: -470 to 40 mm (510 mm), Y: -60 to 100 mm (160 mm), Z: -70 to 70 mm (140 mm)
# Grid: (128, 48, 48) → dims-1 = (127, 47, 47)
sx = 510.0 / 127
sy = 160.0 / 47
sz = 140.0 / 47
print('Approx voxel spacings (mm): X={:.2f} Y={:.2f} Z={:.2f}'.format(sx, sy, sz))
