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

	# weighted average function's denom
	cal_denominator_values = (-2 * tau * tau)

	# determining the appropriate weights for a point
	i = 0

	while(i < cal_X_val_count.shape[0]):

		W[i,i] = np.exp(np.dot((cal_X_val_count[i]-con_curre_data), (cal_X_val_count[i]-con_curre_data).T)/cal_denominator_values)

		i += 1
		
	
	return W


def cal_weight_lineareg():

	# making a numpy array without any data for predictions
	cal_Y_predction_values = np.zeros(y.shape[0])

	cal_err = []

	# analyzing the train dataset's points' weights and predictions.

	cal_poin = 0

	while(cal_poin < X.shape[0]):

		# data (X + 1)
		cal_X_val_count = np.append(X.reshape(X.shape[0],1), np.ones(X.shape[0]).reshape(X.shape[0],1), axis=1)

		# currently available training set datapoint
		con_curre_data = np.array([X[cal_poin], 1])

		# changing the weighted matrix's values
		W = cal_weight_values(con_curre_data, cal_X_val_count)

		# determining theta
		cal_theta_values = np.linalg.pinv(cal_X_val_count.T*(W * cal_X_val_count))*(cal_X_val_count.T*(W * y.reshape(y.shape[0],1)))

		# predicting using regression analysis

		cal_Y_predction_values[cal_poin] = np.dot(con_curre_data, cal_theta_values)
		

		# error calculation
		cal_error_predic = (cal_Y_predction_values[cal_poin] - y[cal_poin]) ** 2
		cal_err.append(cal_error_predic)

		cal_poin += 1

	
	# Locally weighted linear regression function solved.
	prediction_X_count = X.copy() 
	#np.linspace(-3, 3, y.shape[0]) 

	prediction_Y_count = np.zeros(prediction_X_count.shape[0])
	
	cal_poin = 0


	while(cal_poin < prediction_X_count.shape[0]):

		cal_X_val_count = np.append(prediction_X_count.reshape(prediction_X_count.shape[0],1), np.ones(prediction_X_count.shape[0]).reshape(prediction_X_count.shape[0],1), axis=1)

		con_curre_data = np.array([prediction_X_count[cal_poin], 1])

		W = cal_weight_values(con_curre_data, cal_X_val_count)

		cal_theta_values = np.linalg.pinv(cal_X_val_count.T*(W * cal_X_val_count))*(cal_X_val_count.T*(W * y.reshape(y.shape[0],1)))

		pred_count = np.dot(con_curre_data, cal_theta_values)

		prediction_Y_count[cal_poin] = pred_count

		cal_poin += 1

	# Plot for training datapoints and the locally weighted regression function
	#plt.scatter(X, y, label='Training data')   ## Uncomment to print Training datapoints.
	plt.scatter(prediction_X_count, prediction_Y_count)
	plt.title("Locally Weighted Linear Regression")
	plt.legend(loc='upper left')
	plt.grid()
	#plt.savefig("Q2_ABFunctionGraphWithcal_error_predic.png")  ## Uncomment to save the image
	plt.show()
	
		

def main():
	print('START Q2_AB\n')
	'''
	Start writing your code here
	'''
	cal_weight_lineareg()
	print('END Q2_AB\n')


inserting_data = './datasets/Q1_B_train.txt'
data = readFile(inserting_data)
# Inputs:
X = np.asarray([item[0] for item in data], dtype=float)

# Targets
y = np.asarray([item[1] for item in data], dtype=float)
assert X.shape==y.shape, "Shapes of Inputs and Targets are not matching..." 



if __name__ == "__main__":
    main()
