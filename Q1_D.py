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
def Line_Regres_Test(K=10, D=6):
	
   # consider Learning rate.
	L = 0.01


	# considering number of iterations.
	iterations=200


	# Total inputs in dataset.
	n = X_train.shape[0]

	# In the Matrix storing computed errors.
    #using random of numpy
	cal_test_error = np.zeros((K,D+1))

	# 'generatePlots' function call
	fig_plot = []

	# Iterating K from 1 to 10.
	for k in range(1,K+1):
		k_index = []

		print("For K = " + str(k))

		# Iterate on D from 0 to 6.
		for d in range(0, D+1):
			X1 = []
			Y1 = []

			val = -3

			increment = 0.1

			# co-efficients, theta are stored 
			cal_theta_value = np.zeros(d+1)

			# predictions  are stored 
			cal_Y_train_values = np.zeros(y_train.shape[0])

			cal_Y_test_values = np.zeros(y_test.shape[0])
			

			#range for loop add values to x1
			for _ in range(61):

				val += increment

				X1.append(round(val, 2))


			# training a regression model.
			for _ in range(iterations):

				#for cal_point in range(X_train.shape[0]):

				cal_point = 0

				while(cal_point<X_train.shape[0]):	

					# Defining place holder for the function and the constant
					cal_non_lin_equation = 0
					cal_a0_values = 1

					# To create the function based on "d" values, iterate over "d" values.
					#Non-Linear equation


					i = 0

					while(i<d+1):

						cal_non_lin_equation = cal_non_lin_equation+ cal_theta_value[i] * (math.sin(k * i * X_train[cal_point]) * math.sin(k * i * X_train[cal_point]))
						
						i += 1
					
					# Predction
					cal_Y_train_values[cal_point] = cal_a0_values + cal_non_lin_equation

					cal_point+=1

				# Gradient descent is used.
				cal_decent_values_a0 = (-2/n) * sum(y_train - cal_Y_train_values)

				cal_a0_values -= L * cal_decent_values_a0

				# Updating theta values 
				j = 0

				while(j<len(cal_theta_value)):

					D_cal_theta_value = (-2/n) * sum(X_train * (y_train - cal_Y_train_values))

					cal_theta_value[j] -= L * D_cal_theta_value

					j+=1

			# Ccalculate mean squared error on test data
            
			cal_point = 0
			while(cal_point<X_test.shape[0]):
				cal_non_lin_equationTest = 0
				
				i=0 
				while(i<d+1):
					cal_non_lin_equationTest += cal_theta_value[i] * (math.sin(k * i * X_test[cal_point]) * math.sin(k * i * X_test[cal_point]))
					i+=1

				# Save all test predictions
				cal_Y_test_values[cal_point] = cal_a0_values + cal_non_lin_equationTest
				cal_point+=1

			cal_test_error[k-1][d] = round(np.square(y_test-cal_Y_test_values).mean(), 3)

			print("For d = " + str(d) + ' MSE = ' + str(cal_test_error[k-1][d]))
		
			X1_len = len(X1)

			x1 = 0

			while(x1<X1_len):

				cal_non_lin_equation = 0

				for i in range(d+1):

					cal_non_lin_equation += cal_theta_value[i] * (math.sin(k * i * X1[x1]) * math.sin(k * i * X1[x1]))

				Y1.append(cal_a0_values + cal_non_lin_equation)

				x1+=1

			k_index.append({'X' : X1, 'Y' : Y1})

		fig_plot.append(k_index)

	# build plots 
	for k_idx, k_plot in enumerate(fig_plot):

        # calling createPlots 
		createPlots(k_idx, k_plot)

	k_best = np.where(cal_test_error==np.amin(cal_test_error))[0][0]+1

	d_best = np.where(cal_test_error==np.amin(cal_test_error))[1][0]
	
	print("Least Test Error: ", np.amin(cal_test_error))
	print("Least Test Error occurred when 'k': ", str(k_best) + ' d: ' + str(d_best))


# generate plots for K and D
def createPlots(k_idx, plotItem):

	# Create Sine function plot using Matplotlib 
	fig = plt.figure()
	for idx, item in enumerate(plotItem):
		plt.plot(item["X"], item["Y"], 'o-', label=str(idx))
	plt.scatter(X_train, y_train, color = '#88c999')
	plt.title("Trigometric Sine Function for K: " + str(k_idx+1))
	plt.xlabel('X', fontsize=18)
	plt.ylabel('Y', fontsize=16)
	plt.legend(loc='upper left')
	#plt.savefig("Q1D_K" + str(k_idx+1) + ".png")
	plt.show()



def main():
	print('START Q1_D\n')
	'''
	Start writing your code here
	'''
	Line_Regres_Test()
	print('END Q1_D\n')


training_file = './datasets/Q1_B_train.txt'
test_file = './datasets/Q1_C_test.txt'
data_train = readFile(training_file)
data_test = readFile(test_file)

# Train Inputs of first 20 datapoints:
X_train = np.asarray([item[0] for item in data_train[:20]], dtype=float)
# Train Targets of first 20 datapoints
y_train = np.asarray([item[1] for item in data_train[:20]], dtype=float)

# Test Inputs:
X_test = np.asarray([item[0] for item in data_test], dtype=float)
# Test Targets
y_test = np.asarray([item[1] for item in data_test], dtype=float)

assert X_train.shape==y_train.shape, "Shapes of Train Inputs and Train Targets are not matching..." 
assert X_test.shape==y_test.shape, "Shapes of Test Inputs and Test Targets are not matching..." 


if __name__ == "__main__":
    main()

"""
On repeating the experiments ,least test error is 0.261, which occurs when k values is 5 and d value is 6 ,Its is said as a best prediction.
#we see the mean square error to be found  in first 20 elements of the data , where there is no difference between 1B, 1D
"""