import os
from time import time

start_time = time()

print "Executing Naive"
os.system("cd \"Naive Decision Tree\" && python evaluator.py")
naive_completed = time()
print "time taken", naive_completed - start_time

print "Using overfitting prevention methods"
print "Executing part A"
os.system("cd \"Decision Tree with Overfitting Prevention Methods\" && cd a && python evaluator.py")
parta_time = time()
print "time taken", parta_time - naive_completed

print "Executing part B"
os.system("cd \"Decision Tree with Overfitting Prevention Methods\" && cd b && python evaluator.py")
partb_time = time()
print "time taken", partb_time - parta_time

print "Executing part C"
os.system("cd \"Decision Tree with Overfitting Prevention Methods\" && cd c && python evaluator.py")
partc_time = time()
print "time taken", partc_time - partb_time

print "Executing graph part"
os.system("cd \"Decision Tree with Overfitting Prevention Methods\" && cd graphs && python graphs.py")
graph_time = time()
print "time taken", graph_time - partc_time

print "Executing Beam Search"
os.system("cd \"Beam Search over Decision Tree\" && python evaluator.py")
beam_time = time()
print "time taken", beam_time - graph_time
