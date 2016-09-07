import csv
import numpy
from scipy import stats

def normalize(gd):
	gd = stats.zscore(gd, axis=0)
	'''
	mini = gd.min(axis=0)
	maxi = gd.max(axis=0)
	diff = maxi - mini
	gd -= mini
	gd /= diff
	'''
	return gd

def preprocess(input_file, output_dir, fname):

	with open(input_file,"rU") as f:
		num_lines = sum(1 for line in f if line != "\n")
	with open(input_file,"rU") as f:
		line = f.readline()
		print line
		features = len(line.split(',')) - 1
	classdict = {}

	points = numpy.zeros(shape=(num_lines, features+1))
	print num_lines, features
	with open(input_file,"rU") as f:
		reader = csv.reader(f)
		entry = -1
		for row in reader:
			print row
			if row:
				entry += 1
				for i in range(features):
					points[entry][i] = float(row[i])
				if row[features] in classdict:
					points[entry][features] = classdict[row[features]]
				else:
					d_len = len(classdict)
					classdict[row[features]] = d_len
					points[entry][features] = d_len
	print points
	data = points[:,0:features]
	category = points[:,features:features+1]
	data = normalize(data)
	print numpy.hstack([data, category])
	#numpy.savetxt(output_dir+"/"+fname+".data", numpy.hstack([data, category]), delimiter=",")
	with open(output_dir+"/"+fname+".info","w") as f:
		f.write("points "+str(num_lines)+"\n")
		f.write("features "+str(features)+"\n")
		f.write("categories "+str(len(classdict))+"\n")
	with open(output_dir+"/"+fname+".dict", 'wb') as f:
		writer = csv.writer(f)
		for key in classdict:
			value = classdict[key]
			print key,value
			writer.writerow([key, value])


if __name__ == "__main__":
	preprocess("../data/car/car.csv", "../cleaned/car", "car")
