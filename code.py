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

    confusionMatrix = np.zeros((10, 10))
    for c in range(10):
        print("Training Class " + str(c))
        # handling targets
        startingTarget = 240 * c  # 1*0, 1*1, 1*2....etc
        target = np.full(2400, -1)
        target[startingTarget:startingTarget + 240] = 1

        for epoch in range(500):
            for img in range(0, 2400):
                currentImage = images[img]
                if ((1 if (np.dot(weights[c], currentImage) >= 0) else -1) != target[img]):
                    weights[c] = weights[c] + n * currentImage * target[img]

    for i in range(1, 201):
        testImg = plt.imread('./Test/' + str(i) + '.jpg')
        testImg = np.append(testImg.flatten(), 1)
        dot = np.dot(weights, testImg)
        index = np.argmax(dot)
        confusionMatrix[((i - 1) // 20)][index] = confusionMatrix[((i - 1) // 20)][index] + 1


    print(confusionMatrix)
    # plt.imshow(confusionMatrix)
    # plt.savefig("./Confusion-0.jpg")
    # plt.show()





if __name__ == "__main__":

    # images = loadTrainingSet()
    # print(images[0, 10])
    perceptron()
    # print(configureInitialWeightMatrix())

    weight = [1, 2, 1]
    weight = np.array(weight)








