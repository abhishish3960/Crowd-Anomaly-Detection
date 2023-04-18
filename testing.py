import numpy as np
import cv2
def square(a):
    return (a**2)

def diff(l):
    return (l[0] - l[1])
def showUnusualActivities(unusual, vid, noOfRows, noOfCols, n):
   
    unusualFrames = unusual.keys()
    
    print(unusualFrames)
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    
    rows, cols = frame.shape[0], frame.shape[1]
    rowLength = int(rows/(int(noOfRows/n)))
    colLength = int(cols/(int(noOfCols/n)))
    print("Block Size ",(rowLength,colLength))
    count = 0
    
    while 1:
        print(count)
        ret, uFrame = cap.read()
        
        if(count in unusualFrames):
            if (ret == False):
                break
            for blockNum in unusual[count]:
                print(blockNum)
                x1 = blockNum[1] * rowLength
                y1 = blockNum[0] * colLength
                x2 = (blockNum[1]+1) * rowLength
                y2 = (blockNum[0]+1) * colLength
                
            print("Unusual frame number ",str(count))
      
            
      
        if(count == 622):
            break
        
        count += 1
def constructMinDistMatrix(megaBlockMotInfVal,codewords, noOfRows, noOfCols, vid):
    #threshold = 2.1874939946e-21
    #threshold = 0.00196777849633
    #threshold = 8.82926005091e-05
    #threshold = 7.39718222289e-14
    #threshold = 8.82926005091e-05
    #threshold = 0.0080168593265873295
    #threshold = 0.00511863986892
    #------------------------------------#
    threshold = 5.83682407063e-05
    #threshold = 3.37029584538e-07
    #------------------------------------#
    #threshold = 2.63426664698e-06
    #threshold = 1.91130257263e-08
    
    #threshold = 0.0012675861679
    #threshold = 1.01827939172e-05
    n = 2
    minDistMatrix = np.zeros((len(megaBlockMotInfVal[0][0]),(int(noOfRows/n)),(int(noOfCols/n))))
    for index,val in np.ndenumerate(megaBlockMotInfVal[...,0]):
        eucledianDist = []
        for codeword in codewords[index[0]][index[1]]:
            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]),list(codeword)]
          
            dist = np.linalg.norm(megaBlockMotInfVal[index[0]][index[1]][index[2]]-codeword)
            
            eucDist = (sum(map(square,map(diff,zip(*temp)))))**0.5
            
            eucledianDist.append(eucDist)
            
        minDistMatrix[index[2]][index[0]][index[1]] = min(eucledianDist)
    unusual = {}
    for i in range(len(minDistMatrix)):
        if(np.amax(minDistMatrix[i]) > threshold):
            unusual[i] = []
            for index,val in np.ndenumerate(minDistMatrix[i]):
                
                if(val > threshold):
                        unusual[i].append((index[0],index[1]))
    print(unusual)
    showUnusualActivities(unusual, vid, noOfRows, noOfCols, n)
    
def test_video(vid):
    
    print("Test video ", vid)
    MotionInfOfFrames, rows, cols = getMotionInfuenceMap(vid)
   
    megaBlockMotInfVal = createMegaBlocks(MotionInfOfFrames, rows, cols)
   
    np.save(r"/content/drive/MyDrive/btp/Dataset/megaBlockMotInfVal_set1_p1_test_20-20_k5.npy.npy",megaBlockMotInfVal)
    codewords = np.load(r"/content/drive/MyDrive/btp/Dataset/codewords_set1_p1_train_40-40_k5.npy")
    print("codewords",codewords)
    listOfUnusualFrames = constructMinDistMatrix(megaBlockMotInfVal,codewords,rows, cols, vid)
    return
    
if __name__ == '__main__':
   
    testSet = [r"/content/drive/MyDrive/btp/Normal_Abnormal_Crowd/Abnormal Crowds/1183-88_l.mov"]
    for video in testSet:
        test_video(video)
    print("Done")