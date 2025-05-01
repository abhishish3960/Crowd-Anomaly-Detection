import cv2
import numpy as np


def motionInfuenceGenerator(videoPath, block_size=(20, 20), resize_to=(320, 240),
                             pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                             poly_n=5, poly_sigma=1.2, flags=0):
    """
    Generate per-frame block-wise motion-influence features from a video.

    Parameters
    ----------
    videoPath : str
        Path to the input video file.
    block_size : tuple(int, int), optional
        (block_height, block_width) for feature aggregation.
    resize_to : tuple(int, int), optional
        (width, height) to resize video frames before processing.
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags :
        Parameters for cv2.calcOpticalFlowFarneback.

    Returns
    -------
    motion_maps : list of np.ndarray
        Each element has shape (nH, nW, 2), containing mean [magnitude, angle]
        per block for one frame transition.
    nH, nW : int
        Number of blocks along the vertical and horizontal dimensions.
    """
    cap = cv2.VideoCapture(videoPath)
    ret, prev = cap.read()
    if not ret:
        raise IOError(f"Cannot read video {videoPath}")

    if resize_to:
        prev = cv2.resize(prev, resize_to)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    fb_params = dict(pyr_scale=pyr_scale, levels=levels, winsize=winsize,
                     iterations=iterations, poly_n=poly_n, poly_sigma=poly_sigma,
                     flags=flags)
    motion_maps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize_to:
            frame = cv2.resize(frame, resize_to)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # block-wise feature extraction
        features, centres, _ = calcOptFlowOfBlocks(mag, ang, block_size)
        motion_maps.append(features)

        prev_gray = gray

    cap.release()

    if not motion_maps:
        raise ValueError("No frames processed: video may be too short.")

    # derive block-grid dimensions
    nH, nW = motion_maps[0].shape[:2]
    return motion_maps, nH, nW
