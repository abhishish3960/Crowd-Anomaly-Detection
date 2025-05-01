# TRAINING
import numpy as np
from sklearn.cluster import KMeans
import os
import glob
import json

def compute_thresholds(centers, alpha=3.0):
    mu = centers.mean(axis=0)
    dists = [np.linalg.norm(c - mu) for c in centers]
    return float(np.mean(dists) + alpha * np.std(dists))

if __name__ == "__main__":
    trainingSet = glob.glob("/content/drive/MyDrive/btp/Normal_Abnormal_Crowd/Normal Crowds/*.mov")
    save_dir = "/content/drive/MyDrive/btp/Dataset"
    os.makedirs(save_dir, exist_ok=True)

    codebook_folder = os.path.join(save_dir, "codebooks_normal_set")
    threshold_folder = os.path.join(save_dir, "thresholds_normal_set")
    megabook_folder = os.path.join(save_dir, "megablock_normal_set")

    os.makedirs(codebook_folder, exist_ok=True)
    os.makedirs(threshold_folder, exist_ok=True)
    os.makedirs(megabook_folder, exist_ok=True)

    for idx, video in enumerate(trainingSet):
        print(f"Training on {video}")

        maps, R, C = motionInfuenceGenerator(video,resize_to=(320, 240))
        mega = createMegaBlocks(maps, R, C, grid_size=2)
        mb_rows, mb_cols, T, dim = mega.shape

        clusters = 5
        codebooks = np.zeros((mb_rows, mb_cols, clusters, dim), dtype=np.float32)
        thresholds = np.zeros((mb_rows, mb_cols), dtype=np.float32)

        for i in range(mb_rows):
            for j in range(mb_cols):
                data = mega[i, j].reshape(-1, dim)
                kmeans = KMeans(n_clusters=clusters, init='k-means++', n_init=10, random_state=42).fit(data)
                codebooks[i, j] = kmeans.cluster_centers_
                thresholds[i, j] = compute_thresholds(kmeans.cluster_centers_)

        np.save(os.path.join(codebook_folder, f"codewords_normal_{idx}.npy"), codebooks)
        np.save(os.path.join(megabook_folder, f"megaBlockMotInfVal_set1_p1_train_{idx}.npy"), mega)
        np.save(os.path.join(threshold_folder, f"thresholds_normal_{idx}.npy"), thresholds)

        print(f"Saved model for video {idx}")

    print("Training complete.")
