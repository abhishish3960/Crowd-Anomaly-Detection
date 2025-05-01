import cv2
import numpy as np


def calcOptFlowOfBlocks(mag, angle, block_size=(20, 20)):
    """
    Compute block-wise mean optical-flow features and block centers.

    Parameters
    ----------
    mag : np.ndarray, shape (H, W)
        Optical-flow magnitude map.
    angle : np.ndarray, shape (H, W)
        Optical-flow angle map.
    block_size : tuple of int, optional
        (block_height, block_width), default (20, 20).

    Returns
    -------
    features : np.ndarray, shape (nH, nW, 2)
        Mean [magnitude, angle] per block.
    centres : np.ndarray, shape (nH, nW, 2)
        (row, col) center coordinates for each block.
    block_size : tuple of int
        Echoes the input block_size for reference.
    """
    # Smooth magnitude to reduce noise
    mag_smooth = cv2.GaussianBlur(mag, (5, 5), 0)

    H, W = mag_smooth.shape
    bh, bw = block_size
    nH = H // bh
    nW = W // bw

    features = np.zeros((nH, nW, 2), dtype=np.float32)
    centres  = np.zeros((nH, nW, 2), dtype=np.float32)

    # Aggregate mean magnitude & angle per block
    for i in range(nH):
        for j in range(nW):
            r0, r1 = i * bh, (i + 1) * bh
            c0, c1 = j * bw, (j + 1) * bw
            block_mag = mag_smooth[r0:r1, c0:c1]
            block_ang = angle    [r0:r1, c0:c1]

            features[i, j, 0] = block_mag.mean()
            features[i, j, 1] = block_ang.mean()

            centres[i, j] = (r0 + bh / 2, c0 + bw / 2)

    return features, centres, block_size
