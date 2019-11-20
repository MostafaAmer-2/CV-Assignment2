import numpy as np
import math
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

        # handling targets
        startingTarget = 240 * c  # 1*0, 1*1, 1*2....etc
        target = np.full(2400, -1)
        target[startingTarget:startingTarget + 240] = 1

        for e in range(500):
            for img in range(0, 2400):
                currentImage = images[img]
                if ((1 if (np.dot(currentImage, weights[c]) >= 0) else -1) != target[img]):
                    weights[c] = weights[c] + n * target[img] * currentImage

        for i in range(1,201):
            testImg = plt.imread('./Test/'+str(i)+'.jpg')
            testImg = np.append(testImg.flatten(), 1)
            confusionMatrix[c][((i-1)//20)] = confusionMatrix[c][((i-1)//20)]+1 if(np.dot(weights[c], testImg) >= 0) else confusionMatrix[c][((i-1)//20)]

    print(confusionMatrix)

def naive():
    images = loadTrainingSet()
    
    mews = np.zeros([10,784])
    standard_div = np.zeros([10,784])
    for c in range(10):
        mew_c = np.zeros(784)
        for i in range( c*240 , c*240+240):
            mew_c = mew_c + np.true_divide(images[i,:-1],255)
        mews[c] = np.true_divide(mew_c,240)

    for c in range(10):
        std_c = np.zeros(784)
        for i in range( c*240 , c*240+240):
            std_c = std_c + ( pow(np.true_divide(images[i,:-1],255) - mews[c],2) )
        standard_div[c] = np.true_divide( std_c,240 )

    return mews, standard_div
    
def naive_test(filepath):
    mews, standard_div = naive()
    img = plt.imread(filepath)
    img2 = img.flatten()
    img3 = np.append(img2, 1)
    img3 = np.true_divide(img3[:-1],255)

    probs = np.ones(10)
    for c in range(10):
        for i in range(784):
            std_div = 0.1
            if(standard_div[c,i]>0.1):
                std_div = standard_div[c,i]
            gaussian = gaussian_eq(img3[i], mews[c,i], std_div)
            probs[c] = probs[c]*gaussian
    
    print(np.argmax(probs))

    
def gaussian_eq(n, mew, std_div):
    return 1/math.sqrt(2*math.pi*std_div)*math.exp(- ( pow(n - mew,2) ) / ( 2*std_div ) )

if __name__ == "__main__":

    # images = loadTrainingSet()
    # print(images[0, 10])
    # perceptron()
    naive_test("./Test/142.jpg")
    # print(configureInitialWeightMatrix())







