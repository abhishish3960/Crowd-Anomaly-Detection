import numpy as np
import os
import glob
from scipy.spatial import distance
import json


# Function to load thresholds from saved files
def load_thresholds(threshold_folder, video_idx):
    threshold_file = os.path.join(threshold_folder, f"thresholds_normal_{video_idx}.npy")
    if not os.path.exists(threshold_file):
        raise FileNotFoundError(f"Threshold file not found: {threshold_file}")
    return np.load(threshold_file)


def detect_anomalies(mega, codebooks_list, thresholds, alpha=3.0):
    mb_rows, mb_cols, T, dim = mega.shape
    anomalies = []

    for t in range(T):
        for i in range(mb_rows):
            for j in range(mb_cols):
                feat = mega[i, j, t].reshape(-1)

                # Compare with all codebooks and take min Euclidean distance
                dists = []
                for codebooks in codebooks_list:
                    centers = codebooks[i, j]
                    eucl_dists = np.linalg.norm(centers - feat, axis=1)
                    d_min = np.min(eucl_dists)

                    # Get threshold for the current block from the loaded thresholds
                    thr = thresholds[i, j]

                    dists.append((d_min, thr))

                # Use the tightest threshold and smallest distance
                d_min, thr_min = min(dists, key=lambda x: x[0])
                print(f"Frame {t}, Block ({i},{j}): Euclidean Distance = {d_min:.4f}, Threshold = {thr_min:.4f}")

                if d_min > thr_min:
                    anomalies.append((t, i, j))

    return anomalies


if __name__ == "__main__":

    testSet = [
        "/content/drive/MyDrive/btp/Normal_Abnormal_Crowd/Abnormal Crowds/263C044_060_c.mov"
    ]
    codebook_folder = "/content/drive/MyDrive/btp/Dataset/codebooks_normal_set"
    threshold_folder = "/content/drive/MyDrive/btp/Dataset/thresholds_normal_set"
    save_path = "/content/drive/MyDrive/btp/Dataset/megaBlockMotInfVal_set1_p1_test_20-20_k5.npy"

    # Step 1: Load codebooks
    codebook_files = sorted(glob.glob(os.path.join(codebook_folder, "*.npy")))
    if not codebook_files:
        raise FileNotFoundError(f"No codebooks found in: {codebook_folder}")

    codebooks_list = [np.load(f) for f in codebook_files]

    for idx, video in enumerate(testSet):
        print("Testing video:", video)

        # Step 2: Generate mega blocks from video
        maps, R, C = motionInfuenceGenerator(video,resize_to=(320, 240))
        mega = createMegaBlocks(maps, R, C, grid_size=2)
        np.save(save_path, mega)

        # Step 3: Load thresholds for the corresponding video index
        thresholds = load_thresholds(threshold_folder, idx)

        # Step 4: Filter codebooks by shape
        codebooks_list = [cb for cb in codebooks_list if cb.shape[0:2] == mega.shape[0:2]]
        if not codebooks_list:
            raise ValueError("No codebooks matched the test video's mega-block grid size.")

        # Step 5: Detect anomalies
        anomalies = detect_anomalies(mega, codebooks_list, thresholds, alpha=3.0)
        print("Detected anomalies:", anomalies)

        # Automatically visualize
        visualize_anomalies_with_heatmap(
            video_path=video,
            anomalies=anomalies,
            R=R,
            C=C,
            grid_size=2,
            block_size=(20, 20),
            output_path="/content/drive/MyDrive/btp/Dataset/heatmap_output1.mp4",
            display_live=True
        )

    print("All done!")

    print("Done")
