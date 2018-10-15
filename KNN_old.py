import random
from datetime import datetime
import pandas as pd
import operator
import math

class KNN:
    def __init__(self, file_name, attributes, target, k):
        self.file_name = file_name
        self.full_data = []
        self.training_data = []
        self.test_data = []
        self.attributes = attributes
        self.target = target
        self.features = {}
        self.attrValDict = {}
        self.attrValList = {}
        self.makeDistanceMatrixHash = {}

        self.binary_training_data = []
        self.binary_test_data = []
        self.k = k



    def makeDistanceMatrix(self):

        pass


    def execute(self,flag):
        self.read_file()
        self.readAttrList()
        self.convert_to_binary(self.training_data, 'training')
        self.convert_to_binary(self.test_data, 'test')
        actual=[]
        predictions = []

        for x in range(len(self.binary_test_data)):
            neighbors = self.getNeighbors(self.binary_training_data, self.binary_test_data[x], self.k)
            result = self.getResponse(neighbors)
            predictions.append(result)
            actual.append(self.test_data[x][-1])
        accuracy = self.getAccuracy(self.test_data, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')

        if flag:
            y_actu = pd.Series(actual, name='Actual')
            y_pred = pd.Series(predictions, name='Predicted')
            df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
            print("Confusion Matrix for the 100th Sample:")
            print(df_confusion)

        return (100 - accuracy)


    def getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance[0])
        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance[0], trainingSet[x][0], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors


    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response == [1, 0, 0, 0]:
                response = 'unacc'
            elif response == [0, 1, 0, 0]:
                response = 'acc'
            elif response == [0, 0, 1, 0]:
                response = 'good'
            elif response == [0, 0, 0, 1]:
                response = 'vgood'

            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        #sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)

        maximum = max(classVotes, key=classVotes.get)
        return maximum

    def getAccuracy(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0





    def euclideanDistance(self, instance1, instance2, length):
        distance = 0

        for x in range(length):
            distance += pow((int(instance1[x]) - int(instance2[x])), 2)

        #return math.sqrt(distance)
        return distance




    # converts nominal data to binary data
    def convert_to_binary(self, data, flag):
        # iterate through every instance
        for instance in data:
            # pass the instance to the helper function convert() which returns the instance as binary data
            if flag == 'training':
                self.binary_training_data.append(self.convert(instance))
            elif flag == 'test':
                self.binary_test_data.append(self.convert(instance))




    # helper function for convert_to_binary()
    def convert(self, instance):
        new_list = []
        final_list = []
        binary = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

        for i, attribute in enumerate(self.attributes):
            pos = self.attrValDict[attribute][instance[i]]
            binary[i][pos] = 1
        for i, each in enumerate(binary):
            if i < len(binary) - 1:
                new_list = new_list + each
            else:
                new_list = [int(x) for x in new_list]
                each = [int(x) for x in each]
                final_list.append(new_list)
                final_list.append(each)

        return final_list


    def readAttrList(self):
        with open("cardaten/attrList.txt", "r") as file:
            for line in file:
                line = line.strip("\r\n\t")
                key, val = (line.split(':'))
                val_list = val.split(",")
                new_list = {}
                self.attrValList[key] = val_list
                for i, l in enumerate(val_list):
                    new_list[l] = i
                self.attrValDict[key] = new_list


    def read_file(self):
        with open(self.file_name, "r") as file:
            for line in file:
                line = line.strip("\r\n")
                self.full_data.append(line.split(','))

        # get random indices to use as training data (2/3rd of the file)
        # populate the training data list with the instances in those randomly calculated indices
        # populate the test data list with the remaining instances
        training_data_indices = random.sample(range(0, len(self.full_data)-1), int(2 / 3 * len(self.full_data)))

        for index in training_data_indices:
            # populate the training_data list with the randomly calculated indices
            self.training_data.append(self.full_data[index])

        for instance in self.full_data:
            # add the test data instances to the test data list by checking
            # if the instances are in the training data list; if not, they are test data
            if instance not in self.training_data:
                self.test_data.append(instance)




