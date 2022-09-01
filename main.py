import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from random import randint


class NeuralNetwork:
    def __init__ (self, load_save_file=False):
        #initialise our weights to random values
        #our biases will be 0 to begin
        
        #if the model has already done it's training we can
        #load it's presaved values to speed up the process
        #and avoid retraining
        if load_save_file == False:
            print('creating random weights')
            #first layer
            self.w1 = np.random.normal(size=(15, 784)) * np.sqrt(1./(784))
            self.b1 = np.zeros((15, 1))
            #output layer
            self.w2 = np.random.normal(size=(10, 15)) * np.sqrt(1./(30))
            self.b2 = np.zeros((10, 1))
        else:
            print('loading save file')
            self.w1 = np.loadtxt('presaved_params/weight1.csv', delimiter=',')
            b1 = np.loadtxt('presaved_params/bias1.csv', delimiter=',')
            self.w2 = np.loadtxt('presaved_params/weight2.csv', delimiter=',')
            b2 = np.loadtxt('presaved_params/bias2.csv', delimiter=',')
            
            #the save file loads in the biases as the wrong shape 
            #so we need to reshape them inorder to do the proper
            #calculations
            self.b1 = np.reshape(b1, (15, 1))
            self.b2 = np.reshape(b2, (10, 1))

        
    def softMax(self, Z):
        #uses a soft max function to get the 
        #confidence of each possible digit 
        #the network thinks the digit could be
        exp = np.exp(Z - np.max(Z))
        return exp / exp.sum(axis=0)
        # predictions = np.exp(Z) / sum(np.exp(Z))
        # return predictions

    #first activation function used
    #best performance was 93% percent accuracy
    #over 500 trials with a speed of 153.3s
    def sigmoid (self, Z):
        return 1 / (1+np.exp(-Z))

    def sigmoidDerivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    #second activation functoin used
    #best performance was 94% accuracy
    #over 500 trials with a speed of 154.4s
    def hyperbolicTan(self, Z):
        return np.tanh(Z)

    def hyperbolicTanDerivative(self, Z):
        a = self.hyperbolicTan(Z)
        return 1 - (a * a)

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLUDerivative(self, Z):
         return Z > 0

    def forwardPropogation(self, inputs):
        #feed the inputs through our first layer
        self.z1 = self.w1.dot(inputs) + self.b1
        #uses the tanh function to activate our layer
        self.a1 = self.hyperbolicTan(self.z1)
        #feed our previous activations to our next layer
        self.z2 = self.w2.dot(self.a1) + self.b2
        #get our predictions to a certain confidence
        self.a2 = self.softMax(self.z2)


    #encodes the correct answer into a onehot array 
    def oneHotEncoding(self, labels):
        #create an array of zeros according to how many
        #training data we have
        Y = np.zeros((labels.size, labels.max() + 1))
        #for each example we are turning the correct
        #answer to a 1 and the rest of the wrong answers
        #will be a zero
        Y[np.arange(labels.size), labels] = 1
        return Y.T

    def backwardProp(self, inputs, labels, m):
        Y = self.oneHotEncoding(labels)
        #output layer 
        dZ2 =  2 * (self.a2 - Y)
        dw2 = 1 / m * dZ2.dot(self.a1.T)
        db2 = 1 / m * np.sum(dZ2, 1)
        #first layer
        dZ1 = self.w2.T.dot(dZ2) * self.hyperbolicTanDerivative(self.z1)
        dw1 = 1 / m * dZ1.dot(inputs.T)
        db1 = 1 / m * np.sum(dZ1, 1)

        return dw1, db1, dw2, db2

    def update(self, alpha, dw1, db1, dw2, db2):
        #uses the partial derivatives to update
        #the parameters 
        self.w1 = self.w1 - alpha * dw1
        self.b1 = self.b1 - alpha * np.reshape(db1, (15, 1))
        self.w2 = self.w2 - alpha * dw2
        self.b2 = self.b2 - alpha * np.reshape(db2, (10, 1))

    def accuracy(self, predictions, labels):
        print(predictions, labels)
        return np.sum(predictions == labels) / labels.size

    def getPred(self):
        #returns the index which has the highest 
        #value, this represents which number the
        #model is saying the digit is
        return np.argmax(self.a2, 0)

    def train (self, inputs, labels, m, alpha, trials):
        #train the model for x number of trials
        for i in range(trials):
            #gradient decent
            self.forwardPropogation(inputs)
            dw1, db1, dw2, db2 = self.backwardProp(inputs, labels, m)
            self.update(alpha, dw1, db1, dw2, db2)

            #every 100 trials print out the accuracy
            #to monitor how the model is doing
            if i % 100 == 0:
                print ("trial:", i)
                predictions = self.getPred()
                print(self.accuracy(predictions, labels))
            
    def make_pred(self, image):
        self.forwardPropogation(image)
        prediction = self.getPred()
        return prediction

    #given the test inputs go through and let the model
    #make a prediction
    def test(self, index, test_inputs, test_labels):
        curr = test_inputs[:, index, None]
        prediction = self.make_pred(curr)
        label = test_labels[index]
        print("Prediction: ", prediction)
        print("Label:", label)

        #plot the images so we can see what the
        #model is seeing
        image = curr.reshape((28,28))
        plt.imshow(image, cmap='gray')
        plt.show()

    def saveParams(self):
        #saves the parameters of our current model into 
        #a csv file to reuse the same parameters and avoid
        #retraining after every start up
        weight1 = np.asarray(self.w1)
        bias1 = np.asarray(self.b1)
        weight2 = np.asarray(self.w2)
        bias2 = np.asarray(self.b2)

        np.savetxt('presaved_params/weight1.csv', weight1, delimiter=',')
        np.savetxt('presaved_params/bias1.csv', bias1, delimiter=',')
        np.savetxt('presaved_params/weight2.csv', weight2, delimiter=',')
        np.savetxt('presaved_params/bias2.csv', bias2, delimiter=',')






#create a network
#inputing an argument of true loads in the 
#parameters from the csv files of an 
#already trained model
network = NeuralNetwork(True)

#get the data from the csv
data = pd.read_csv('mnsit_data/mnist_train.csv')
data = np.array(data)
m, n = data.shape

#transpose the data so that we can get
#the labels and the inputs individually
data = data.T

data_labels = data[0]
data_inputs = data[1:n]

#get the test data the same way we got the 
#training data
test_data = pd.read_csv('mnsit_data/mnist_test.csv')
test_data = np.array(test_data)
x, y = test_data.shape

test_data = test_data.T
test_labels = test_data[0]
test_inputs = test_data[1:y]



print('training...')
#train the network with an alpha of 0.1 
#for 500 trials
network.train(data_inputs, data_labels, m, 0.1, 500)

network.train(test_inputs, test_labels, m, 0.1, 250)


print('testing')

#run the model through a test of 10 digits to see how it preforms
for i in range(10):
    index = randint(0, 10000)
    network.test(index, test_inputs, test_labels)



print ('if you would like to save these parameters enter a 1, enter anything else to quit without saving')
print()
save = int(input())

if save == 1:
    network.saveParams()
