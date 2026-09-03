import numpy as np
import math
import os 
import matplotlib.pyplot as plt

files = "SOON-pEMP\Data"
filesToWrite = "code\Data"
resultsTime = []
notSuitable = ["05","13","14","16",]

for file in os.scandir(files):
    output_path = os.path.join(filesToWrite, file.name)
    if file.is_file() and not any(pattern in file.name for pattern in notSuitable):
        with open(output_path, "w") as outputFile:
            outputFile.write("Combined Vectors\n")
            with open(file.path, "r") as csvfile:
                next(csvfile) # Skip header
                previous_timestamp = None
                file_deltas = []
                
                for line in csvfile.readlines():   
                    array = line.strip().split(',')
                    if len(array) == 4:
                        vector = math.sqrt((float(array[1]) ** 2) + (float(array[2]) ** 2) + (float(array[3]) ** 2))
                        outputFile.write(f"{array[3]}\n")
                        
                        current_timestamp = float(array[0])
                        if previous_timestamp is not None:
                            delta = current_timestamp - previous_timestamp
                            file_deltas.append(delta)
                            resultsTime.append(delta)
                        previous_timestamp = current_timestamp
                
print("Total statistics:")
print(f"Number intervals: {len(resultsTime)}")
print(f"Mean delta-t: {np.mean(resultsTime)}"  )
print(f"Standard deviation: {np.std(resultsTime)}" )
print(f"Median: {np.median(resultsTime)}" )
print(f"Minimum: {np.min(resultsTime)}" )
print(f"Maximum: {np.max(resultsTime)}" )

plt.plot(resultsTime)
plt.show()