import numpy as np
import matplotlib.pyplot as plt


# Removing "," and "(", ")" characters from the dataset
def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


# Retrieve data from the file
def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


# Reading file as a numpy array
def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    input_np = np.array(input_data)
    return input_np


# defining computer weighted matrix
def cal_weight_values(con_curre_data, cal_X_val_count): 

	# for defining the locality, Tau
	tau = 0.204

	# Initializing a matrix of identities
	W = np.mat(np.eye(cal_X_val_count.shape[0]))

	# determining the appropriate weights for a point
	cal_denominator_values = (-2 * tau * tau)

	# calculating the corresponding weights for a point
	i = 0

	while(i < cal_X_val_count.shape[0]):

		W[i,i] = np.exp(np.dot((cal_X_val_count[i]-con_curre_data), (cal_X_val_count[i]-con_curre_data).T)/cal_denominator_values)

		i += 1
	
	return W


def cal_weight_lineareg():
    # making a numpy array without any data for predictions
	cal_Y_predction_values = np.zeros(input_y_data_test.shape[0])

	cal_err = []

   # analyzing the train dataset's points' weights and predictions.
	cal_poin = 0

	while(cal_poin < input_X_data_test.shape[0]):

		# data (X + 1)
		cal_X_val_count = np.append(input_X_data_train.reshape(input_X_data_train.shape[0],1), np.ones(input_X_data_train.shape[0]).reshape(input_X_data_train.shape[0],1), axis=1)

		# currently available training set datapoint 
		con_curre_data = np.array([input_X_data_test[cal_poin], 1])

		# changing the weighted matrix's values
		W = cal_weight_values(con_curre_data, cal_X_val_count)

		# determining theta
		cal_theta_values = np.linalg.pinv(cal_X_val_count.T*(W * cal_X_val_count))*(cal_X_val_count.T*(W * input_y_data_train.reshape(input_y_data_train.shape[0],1)))

		# predicting using regression analysis
		cal_Y_predction_values[cal_poin] = np.dot(con_curre_data, cal_theta_values)


		# error calculation
		cal_error_predic = (cal_Y_predction_values[cal_poin] - input_y_data_test[cal_poin]) ** 2
		cal_err.append(cal_error_predic)

		cal_poin += 1
	
	print("Data Size: " + str(input_X_data_train.shape[0]) + ", MSE:", round(np.mean(cal_err), 6))



def main():
	print('START Q2_C\n')
	'''
	Start writing your code here
	'''
	cal_weight_lineareg()
	print('END Q2_C\n')


training_file = './datasets/Q1_B_train.txt'
test_file = './datasets/Q1_C_test.txt'
training_data = readFile(training_file)
test_data = readFile(test_file)
# Inputs of train:
input_X_data_train = np.asarray([item[0] for item in training_data], dtype=float)
# Inputs of test
input_X_data_test = np.asarray([item[0] for item in test_data], dtype=float)

# Targets of train
input_y_data_train= np.asarray([item[1] for item in training_data], dtype=float)
# Targets of test
input_y_data_test= np.asarray([item[1] for item in test_data], dtype=float)

assert input_X_data_train.shape==input_y_data_train.shape, "Shapes of Inputs and Targets on training set are not matching..." 
assert input_X_data_test.shape==input_y_data_test.shape, "Shapes of Inputs and Targets on test set are not matching..." 


if __name__ == "__main__":
    main()
    