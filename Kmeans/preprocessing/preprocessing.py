'''
Preprcessing the car evalution dataset to feed as an input to decision tree.
'''


import csv
import numpy as np
import random

class Preprocess(object):
    def printh(self):
        print "hi"

    def __init__(self, file_dir, dest_dir, dataset_name, new_name):
        num_attrs = []
        with open(file_dir+"types.csv", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "Numerical":
                    num_attrs.append(int(row[1]))

        if 0 in num_attrs:
            num_attrs.remove(0)
        if 1 in num_attrs:
            num_attrs.remove(1)

        num_attrs.append(1)


        data = np.loadtxt(file_dir+dataset_name+".csv", delimiter=',', usecols=num_attrs, dtype=int, skiprows=1)

        new_data = np.zeros(shape=data.shape, dtype = int)
        self.map_it(data, new_data)


        data_to_include = new_data #[random.sample(range(new_data.shape[0]), 1800),:]
        fn = dest_dir+new_name+".csv"
        np.savetxt(fn, data_to_include, delimiter=",", fmt="%d")

    def chunkIt(self, seq, num):
        arr = [len(seq) / num] * num
        k = len(seq) - sum(arr)
        while k > 0:
            arr[k - 1] += 1
            k -= 1
        arr = [i for i in arr if i > 0]
        arr = [0] + arr
        for i in range(1, len(arr)):
            arr[i] += arr[i - 1]
        out = []
        for i in range(0, len(arr) - 1):
            out.append(seq[arr[i]:arr[i + 1]])

        dict = {}
        for part in range(len(out)):
            for point in out[part]:
                dict[point] = part
        return dict

    def map_it(self, data, new_data):
        rows, columns = data.shape
        print rows, columns
        map_dict = {}

        for i in range(columns):
            vals = list(set(data[:,[i]].reshape(-1,)))
            vals.sort()
            map_dict[i] = self.chunkIt(vals, 10)

        for i in range(rows):
            for j in range(columns):
                new_data[i][j] = map_dict[j][data[i][j]]


if __name__ == "__main__":
    preprocessor = Preprocess('..\data\\', "..\cleaned\\", "training", "training")
    preprocessor.printh()
