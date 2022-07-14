'''
EnvironmentPython 3.6,  pyts: 0.11.0, Pandas: 1.0.3
'''
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyts.datasets import load_gunpoint
from pyts.image import MarkovTransitionField
import matplotlib.pyplot as plt
import numpy as np
s = 20
k = 10
## call API
X, _, _, _ = load_gunpoint(return_X_y=True)
mtf = MarkovTransitionField()
fullimage = mtf.transform(X)

# downscale MTF of the time series (without paa) through mean operation
batch = int(len(X[0]) / s)
patch = []
for p in range(s):
    for q in range(s):
        patch.append(np.mean(fullimage[0][p * batch:(p + 1) * batch, q * batch:(q + 1) * batch]))
# reshape
patchimage = np.array(patch).reshape(s, s)

plt.figure()
plt.suptitle('gunpoint_index_' + str(k))
ax1 = plt.subplot(121)
plt.imshow(fullimage[k])
plt.title('full image')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(cax=cax)

ax2 = plt.subplot(122)
plt.imshow(patchimage)
plt.title('MTF with patch average')
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.2)
plt.colorbar(cax=cax)
plt.show()
