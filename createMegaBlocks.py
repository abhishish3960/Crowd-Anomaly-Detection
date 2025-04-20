import numpy as np

def createMegaBlocks(motionInfoOfFrames, num_block_rows, num_block_cols, grid_size=2):
    """
    Group motion-influence block features into mega-blocks for k-means.

    Parameters
    ----------
    motionInfoOfFrames : list of np.ndarray
        Each element is an array of shape (num_block_rows, num_block_cols, feat_dim)
        containing the block-wise [mean_magnitude, mean_angle] for one frame.
    num_block_rows : int
        Number of blocks along the vertical axis.
    num_block_cols : int
        Number of blocks along the horizontal axis.
    grid_size : int, optional
        Number of blocks per mega-block side (default is 2).

    Returns
    -------
    np.ndarray
        Array of shape (num_block_rows//grid_size,
                      num_block_cols//grid_size,
                      T,
                      feat_dim * grid_size * grid_size)
        where T = len(motionInfoOfFrames).
    """
    # Stack frames -> shape (T, R, C, D)
    motion_maps = np.stack(motionInfoOfFrames, axis=0)
    T, R, C, D = motion_maps.shape

    # Ensure divisibility
    if R % grid_size != 0 or C % grid_size != 0:
        raise ValueError(f"Block grid ({R}×{C}) not divisible by grid_size={grid_size}")

    mb_rows = R // grid_size
    mb_cols = C // grid_size
    mega_dim = D * grid_size * grid_size

    # Initialize output
    mega_blocks = np.zeros((mb_rows, mb_cols, T, mega_dim), dtype=np.float32)

    # Fill each mega-block
    for i in range(mb_rows):
        for j in range(mb_cols):
            # Extract the grid_size×grid_size patch across all T frames
            patch = motion_maps[
                :,
                i*grid_size:(i+1)*grid_size,
                j*grid_size:(j+1)*grid_size,
                :
            ]  # shape (T, grid_size, grid_size, D)
            # Flatten local blocks into feature vector
            mega_blocks[i, j] = patch.reshape(T, -1)

    return mega_blocks
