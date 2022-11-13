import numpy as np
import math
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


# Consider K and D values in linear regression function
def Line_Reg_Test(K=10, D=6):
	
    # consider Learning rate.
	L = 0.01

# considering number of iterations.
	iterations=200

	# Learning rate.
	L = 0.01

	# Total inputs in dataset.
	n = X_train.shape[0]

	# In the Matrix storing computed errors.
    #using random of numpy
	test_error_calculate = np.zeros((K,D+1))

	# Stroring the computer Coefficients for the trignometric function in the Matrix.
	#coefficientsMatrix = np.zeros((K,D+1), dtype=object)

	# Iterating K from 1 to 10.
	for k in range(1,K+1):

		print("For K = " + str(k))
		# Iterate on D from 0 to 6.
		for d in range(D+1):
			X1, Y1 = [], []
			val = -3
			increment = 0.1

			# co-efficients, theta are stored in linear regression.
			cal_theta = np.zeros(d+1)

			# predictions are stored in linear regression.
			#Return new Array of given shape and type
			cal_Y_train_val = np.zeros(y_train.shape[0])
			cal_Y_test_val = np.zeros(y_test.shape[0])

			#range for loop add values to x1
			for _ in range(61):
				val += increment
				val = round(val, 2)
				X1.append(val)

			# training a regression model..
			for _ in range(iterations):

				#while cal_poin in range(X_train.shape[0]):
				cal_poin = 0
				while(cal_poin<X_train.shape[0]):

					# store function's and the constant's values
					cal_non_lineq, a0 = 0, 1

					# To create the function based on "d" values, iterate over "d" values.
					#Non-Linear equation
					i=0
					while(i<d+1):
						cal_non_lineq += cal_theta[i] * (math.sin(k * i * X_train[cal_poin]) * math.sin(k * i * X_train[cal_poin]))
						i+=1

					
					# Predction
					cal_Y_train_val[cal_poin] = a0 + cal_non_lineq
					cal_poin+=1

				# Gradient descent is used.
				cal_desce_valuea0 = (-2/n) * sum(y_train - cal_Y_train_val)
				a0 = a0 - L * cal_desce_valuea0

				# Updating theta values 
				j = 0
				while(j<len(cal_theta)):
					D_cal_theta = (-2/n) * sum(X_train * (y_train - cal_Y_train_val))
					cal_theta[j] = cal_theta[j] - L * D_cal_theta
					j+=1


			# Compute mean squared error on test data

			cal_poin = 0
			while(cal_poin<X_test.shape[0]):
				cal_non_lineqTest = 0
				
				i=0 
				while(i<d+1):
					cal_non_lineqTest += cal_theta[i] * (math.sin(k * i * X_test[cal_poin]) * math.sin(k * i * X_test[cal_poin]))
					i+=1

				# Save all test predictions
				cal_Y_test_val[cal_poin] = a0 + cal_non_lineqTest
				cal_poin+=1

			test_error_calculate[k-1][d] = round(np.square(y_test-cal_Y_test_val).mean(), 3)
			print("For d = " + str(d) + ' MSE = ' + str(test_error_calculate[k-1][d]))
		
			cal_X1_count = len(X1)
			x1 = 0
			while(x1<cal_X1_count):
				cal_non_lineq = 0

				for i in range(d+1):
					cal_non_lineq += cal_theta[i] * (math.sin(k * i * X1[x1]) * math.sin(k * i * X1[x1]))
				Y1.append(a0 + cal_non_lineq)
				x1+=1

	k_best = np.where(test_error_calculate==np.amin(test_error_calculate))[0][0]+1
	d_best = np.where(test_error_calculate==np.amin(test_error_calculate))[1][0]
	print("Least Test Error: ", np.amin(test_error_calculate))
	print("Least Test Error occurred when 'k': ", str(k_best) + ' d: ' + str(d_best))


def main():
	print('START Q1_C\n')
	'''
	Start writing your code here
	'''
	Line_Reg_Test()
	print('END Q1_C\n')


training_file = './datasets/Q1_B_train.txt'
test_file = './datasets/Q1_C_test.txt'
data_train = readFile(training_file)
data_test = readFile(test_file)

# Train Inputs:
X_train = np.asarray([item[0] for item in data_train], dtype=float)
# Train Targets
y_train = np.asarray([item[1] for item in data_train], dtype=float)

# Test Inputs:
X_test = np.asarray([item[0] for item in data_test], dtype=float)
# Test Targets
y_test = np.asarray([item[1] for item in data_test], dtype=float)

assert X_train.shape==y_train.shape, "Shapes of Train Inputs and Train Targets are not matching..." 
assert X_test.shape==y_test.shape, "Shapes of Test Inputs and Test Targets are not matching..." 

if __name__ == "__main__":
    main()

"""

Over fitting occures when there is large data, here by graphs we can say over fitting occures.
on evaluating the functions,least test error is 0.377, which occurs when k values is 2 and function depth value is 4 ,Its is said as a best prediction.
"""