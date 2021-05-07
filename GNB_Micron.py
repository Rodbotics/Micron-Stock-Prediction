# START OF DOCUMENTATION SECTION #

# PROGRAM NAME: Gaussian Naive Bayes (GNB) Classification of Micron Stock Data w/ k-fold cross validation

# SUMMARY OF PROGRAM:
# This program performs a k-fold cross validation procedure of 1 year's worth of Micron stock data. I trained the training set...
# using Naive Bayes Classifier and validated using validation (testing) sets.
# For the training process the amount of parameters trained was eight (mean and standard deviation for every feature for each label)

# PROGRAM DETAILS:
# Class/Label: 2 --> UP (2) or DOWN (1)
# Features: 4 --> Open, High, Low, Close
# Number of folders: 12 (try different k)
# Total trained parameters: 16 (8 parameters per class/label)
# Dataset labels represents whether the stock closes up/down relative to the previous day
# These 2 labels will dictate the stocks predicted trend direction
# Dataset has 4 independent features and 252 datapoints (# of trading days) per feature

# RESULTS:
# After conducting various runs, the average accuracy ranges from ~45.83% to ~52.84%


# END OF DOCUMENTATION #

############################################################################################################################################

# START OF PROGRAM #

from scipy.stats import norm
import numpy as np
import csv

##############
# READING FILE

# Initializing data set as a list
data_set = []

with open('Micron 1Y Stock Data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data_set.append([int(row['Direction']), float(row['Open']), float(row['High']), float(row['Low']),
                         float(row['Close'])])

data_set = np.array(data_set)   # Stores 252x5 matrix. 1st column are labels, 2nd:end are features
# print(data_set, '\n')           # 137 rows are labeled UP (2) and 115 are labeled DOWN (1)

#########################################
# PREPPING FOR K-FOLD CROSS VALIDATION

np.random.shuffle(data_set)                             # Randomly shuffles each row in the original dataset
feature_set = data_set[:, 1:np.shape(data_set)[1]]      # All rows for every feature (columns 1-4) ---> size: [252x4]
label_set = data_set[:, 0].astype(int)                  # All rows for label (column 0) ---> size: [252x1]
# print(feature_set, '\n')
# print(label_set, '\n')
k = 12                                                  # Use 12 folders for k
data_set_split = np.array_split(data_set, k)            # Splitting data set into k-subarrays
# print(data_set_split, '\n')

################################################################
# CALCULATING MEAN AND STANDARD DEVIATION FOR GNB CLASSIFICATION


# Function returns mean and standard deviation from given inputs (feature vectors)
def mean_and_standev(features):
    f_d = np.shape(features)[1]                     # Column vector of 4's ---> [4 4 4].T
    # print(f_d, '\n')
    # print(np.shape(features), '\n')                # (137, 4) & (115, 4) = 137 rows. 4 columns for UP (2), 4 for DOWN (1)
    mean = np.empty(f_d)                            # Initializing an array except with really small numbers
    standev = np.empty(f_d)
    for k in range(f_d):                            # Loops 4 times, 4 times over. Total of 16 times --> 0,1,2,3; 0,1,2,3; 0,1,2,3, 0,1,2,3
        mean[k] = np.average(features[:, k])                # Calculating mean for each feature (1st, 2nd, 3rd, 4th) for both labels
        standev[k] = np.std(features[:, k], ddof=1)         # Calculating std for each feature (1st, 2nd, 3rd, 4th) for both labels

    return mean, standev        # Mean and standard deviation are both 4x4 matrices

#####################################
# GAUSSIAN NAIVE BAYES TRAINING MODEL


# Function returns trained parameters m and s
def training(feature_set, label_set, number_of_label=2):
    f_d = np.shape(feature_set)[1]                      # Stores an int = 4
    # print(range(f_d), '\n')
    m = np.empty([number_of_label, f_d])                # Initializing mean for every label
    s = np.empty([number_of_label, f_d])                # Initializing standard deviation for every label

    # There are 2 labels: UP (2) & DOWN (1)
    for i in range(number_of_label):                    # Loops 2 times for 2 labels . Total of 4 times --> 0,1; 0,1
        features = feature_set[label_set == i + 1]      # 4, 2d arrays of each feature's data pts for each label. (137x4) & (115x4), ...
        mean, standev = mean_and_standev(features=features)     # Calling function "mean_and_standev" and inserting features as input
        m[i, :] = mean           # Mean for each label. 1st row = average for label 1, 2nd row means average for label 2
        s[i, :] = standev        # Standard deviation for each label. 1st row means standev for label 1 (in 3 dimension individually), ...

    return m, s

############################################################
# CALCULATE LIKELIHOOD OF EACH LABEL FOR GNB CLASSIFICATION


# Function that calculates likelihood from given input arguments
def likelihood(mean, standev, sample_vector, label_set):
    f_d = np.shape(sample_vector)                                           # Feature vector dimension
    l_d = np.shape(label_set)[0]                                            # Label vector dimension (252 labels)
    density_array = np.empty([f_d[0], 2])                                   # Array size: [252x2]
    # print(f_d, '\n')
    # print(l_d, '\n')
    # print(density_array, '\n')
    for i in range(f_d[0]):                                                 # Range(0,252)
        for j in range(2):                                                  # Range(0,2)
            density = 1.0                                                   # Initialize density
            prior = np.count_nonzero(label_set == j + 1) / l_d              # Label 1 prior: 137/252, Label 2 prior: 115/252
            for k in range(f_d[1]):
                density *= norm.pdf(sample_vector[i][k], mean[j][k], standev[j][k])     # Calculates probability density function
            density_array[i][j] = density * prior                           # Creates an array of pdf's and priors
    # print(sample_vector)
    # print(prior)
    test = np.argmax(density_array, axis=1) + 1     # Returns the index with the largest value in density array as UP (2) or DOWN (1)
    # print(test)
    return test

####################################################
# CALCULATE THE AVERAGE ACCURACY OF MODELS (12-FOLD)


# Initializing loop counters
correct_matches = 0
validation_sample = 0

for ii in range(k):                                                 # Loops from 0 to k (i.e. 0,1,2,3,4,...,k)
    validation_data_split = data_set_split[ii]                      # Creating validation sets from k-folds (12 folders each size: 21x5)
    feature_set_test = validation_data_split[:, 1:np.shape(validation_data_split)[1]]   # Features of validation set to use as test set (21x4)
    label_set_test = validation_data_split[:, 0].astype(int)                            # Labels of validation set to use as test set (1x21)
    # print(feature_set_test, '\n')
    # print(label_set_test, '\n')
    for i in range(k):                                              # Loops from 0 to k (i.e. 0,1,2,3,4,...,k)
        if ii != i:                                                 # Do the following if indices do not match
            current_data_split = data_set_split[i]  # Training sets( for k = 12 --> 11 other training sets of size: 21x5)
            feature_set = current_data_split[:, 1:np.shape(current_data_split)[1]]      # Features of training set to use for GNB training (21x4)
            label_set = current_data_split[:, 0].astype(int)                            # Labels of training set to use for GNB training (1x21)
            mean_trained, variance_trained = training(feature_set=feature_set, label_set=label_set)     # Calling training function
            probability = likelihood(mean_trained, variance_trained, feature_set_test, label_set)       # Calling likelihood function
            # print(probability)
            for j in range(4):                                      # Loops through all feature columns in dataset --> 4
                validation_sample = validation_sample + 1           # Validation_sample is equal to 1 ---> 2 ---> 3 ---> 4
                if probability[j] == label_set_test[j]:             # Checking if correct predictions = the current label test set
                    print(probability)                              # Prints likelihood
                    correct_matches = correct_matches + 1           # Increment the correct predictions by +1 if condition is met

# Finding average accuracy of the GNB model
print('The average accuracy is: ' + str(round((correct_matches / validation_sample * 100), 2)) + '%')


# END OF PROGRAM #

############################################################################################################################################
