import numpy as np
import csv
import torch

path = "./white_wine_data/wine.csv"

data = np.loadtxt(path, dtype=np.float32, delimiter=";", skiprows=1)
columns = next(csv.reader(open(path), delimiter=";"))

data_tensor = torch.from_numpy(data)

input_data = data_tensor[:, :-1]
target_data = data_tensor[:, -1].long()

one_hot_encoded = torch.zeros(target_data.shape[0], 10)
one_hot_encoded = one_hot_encoded.scatter_(1, target_data.unsqueeze(1), 1.0)


#MANUAL analysis of good wines below

#normalize the data
input_mean = torch.mean(input_data, dim=0)
standard_deviation = torch.var(input_data, dim=0)

normalized_input_data = (input_data - input_mean) / torch.sqrt(standard_deviation)


bad_wines = target_data <= 3

#advanced indexing feature - allows us to use the bool tensor of bad wines to show the bad wines 
# to show the bad wines in the total data
filtered_data = input_data[bad_wines]


#group data

bad = input_data[target_data <=3]
mid = input_data[(target_data > 3) & (target_data < 7)]
good = input_data[target_data > 7]

bad_mean = torch.mean(bad, dim=0)
mid_mean = torch.mean(mid, dim=0)
good_mean = torch.mean(good, dim=0)

for i, args in enumerate(zip(columns, bad_mean, mid_mean, good_mean)):
    print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))
   
sulufur_threshold = 141.83 
sulfur_data = input_data[:, 6]
predicted_indexes = torch.lt(sulfur_data, sulufur_threshold) #compares a tensor -> value or tensor -> tensor element wise


print(predicted_indexes.shape, predicted_indexes.dtype, predicted_indexes.sum())

indexes_of_good_wines = target_data > 5 # returns bool tensor

number_of_matches = torch.sum(indexes_of_good_wines & predicted_indexes).item()
predicted_good_wines = torch.sum(predicted_indexes).item()
actual_good_wines = torch.sum(indexes_of_good_wines).item()

print(number_of_matches, number_of_matches / predicted_good_wines, number_of_matches / actual_good_wines)






