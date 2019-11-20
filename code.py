import numpy as np
import matplotlib.pyplot as plt

def loadTrainingSet():
    images = np.empty([2400, 785], dtype=np.float64)
    for i in range(1, 2401):
        img = plt.imread('./Train/' + str(i) + '.jpg')
        img2 = img.flatten()
        img3 = np.append(img2, 1)
        images[i-1] = img3

    return images

def configureInitialWeightMatrix():
    weights = np.zeros([10, 784])
    ones = np.ones([10, 1])
    return np.hstack((ones, weights))


def perceptron():
    images = loadTrainingSet()
    weights = configureInitialWeightMatrix()
    n = 1

    #handling targets
    startingTarget = 240 * 0  # 1*0, 1*1, 1*2....etc
    target = np.full(2400, -1)
    target[startingTarget:startingTarget + 240] = 1

    for e in range(500):
        for img in range(0, 2400):
            currentImage = images[img]
            if ((1 if (np.dot(currentImage, weights[0]) >= 0) else -1) != target[img]):
                weights[0] = weights[0] + n * target[img] * currentImage

    confusionMatrix_0 = np.zeros((10, 10))
    correct = 0
    false = 0
    for i in range(1,201):
        testImg = plt.imread('./Test/'+str(i)+'.jpg')
        testImg = np.append(testImg.flatten(), 1)
        testingResult = True if(np.dot(weights[0], testImg) >= 0) else False

        if(testingResult):
            correct = correct +1
        else:
            false = false + 1

        # confusionMatrix_0[((i-1)//20), testingResult] += 1

    print("correct:" + str(correct))
    print("false:" + str(false))





if __name__ == "__main__":

    # images = loadTrainingSet()
    # print(images[0, 10])
    perceptron()
    # print(configureInitialWeightMatrix())







