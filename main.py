from KNN import KNN
import sys
from datetime import datetime

if __name__ == "__main__":
    file_name = "cardaten/car.data"
    attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    target = 'acceptability'

    k = int(sys.argv[1])

    sum_of_errors = 0

    print('Creating Distance Matrix... Please Wait for a few minutes')
    knn = KNN(file_name, attributes, target, k)
    knn.read_file()
    knn.readAttrList()
    knn.makeDistanceMatrix()

    for i in range(0, 99):

        print("Running Iteration ", i+1, "...")
        knn.split_data()
        sum_of_errors += knn.execute(False)

    print("Running Iteration ", 100, "...")
    knn.split_data()
    sum_of_errors += knn.execute(True)

    mean_error = sum_of_errors/100
    print("Mean Error : ", mean_error)
