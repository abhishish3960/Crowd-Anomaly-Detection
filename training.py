import numpy as np

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
def train_from_video(vid):
    
    print("Training From ", vid)
    MotionInfOfFrames, rows, cols = getMotionInfuenceMap(vid)
    print("Motion Inf Map", len(MotionInfOfFrames))
    megaBlockMotInfVal = createMegaBlocks(MotionInfOfFrames, rows, cols)
    np.save(r"/content/drive/MyDrive/btp/Dataset/megaBlockMotInfVal_set1_p1_train_40-40_k5.npy",megaBlockMotInfVal)
    print(np.amax(megaBlockMotInfVal))
    print(np.amax(reject_outliers(megaBlockMotInfVal)))
    
    codewords = kmeans(megaBlockMotInfVal)
    np.save(r"/content/drive/MyDrive/btp/Dataset/codewords_set1_p1_train_40-40_k5.npy",codewords)
    print(codewords)
    return
    
if __name__ == '__main__':
   
    trainingSet = [r"/content/drive/MyDrive/btp/Normal_Abnormal_Crowd/Normal Crowds/341-46_l.mov",r"/content/drive/MyDrive/btp/Normal_Abnormal_Crowd/Abnormal Crowds/1183-88_l.mov",]
    for video in trainingSet:
        train_from_video(video)
    print("Done")