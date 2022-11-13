#import statements
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
def Line_Reg(K=10, D=6):

	# consider Learning rate.
	L = 0.01
	
	# considering number of iterations.
	no_of_iterations=200
	
	# Total inputs in dataset.
	n = X.shape[0]

	# In the Matrix storing computed errors.
    #using random of numpy
	error_calculate = np.random.rand(K,D+1)

	#coeffications are sorted in the matrix
	coeffTrigFunct = np.zeros((K,D+1), dtype=object)

	# 'generatePlots' function call
	figure_Plot = []

	# Iterating K from 1 to 10.
	k=1
	while(k<K+1):
		k_index = []
		print("For K = " + str(k))

		# Iterate on D from 0 to 6.
		d=0
		while(d<(D+1)):
			val = -3
			increment = 0.1
			X1 = []
			Y1 = []
			

			# co-efficients, theta are stored in linear regression.
			
			theta_cal = np.zeros(d+1)

			# predictions are stored in linear regression.
			#Return new Array of given shape and type
			Y_cal_pred = np.zeros(y.shape[0])
			
			#range for loop add values to x1
			for _ in range(61):
				val = val + increment
				val = round(val, 2)
				X1.append(val)


			# training a regression model.
			for _ in range(no_of_iterations):
				cal_poin_value = 0
				while(cal_poin_value<X.shape[0]):


					# store function's and the constant's values
					cal_nonLine_eqa = 0
					a0 = 1

					# To create the function based on "d" values, iterate over "d" values.
					#Non-Linear equation
					i = 0
					while(i<d+1):
						cal_nonLine_eqa = cal_nonLine_eqa + theta_cal[i] * (math.sin(k * i * X[cal_poin_value]) * math.sin(k * i * X[cal_poin_value]))
						i = i + 1

					
					# Predction
					Y_cal_pred[cal_poin_value] = a0 + cal_nonLine_eqa
					cal_poin_value+=1

				# Gradient descent is used.
				calcul_Gradescent_a0 = (-2/n) * sum(y - Y_cal_pred)

				a0 -=  L * calcul_Gradescent_a0

				# Updating theta values 
				j = 0

				while(j<len(theta_cal)):

					D_theta_cal = (-2/n) * sum(X * (y - Y_cal_pred))
					theta_cal[j] -= L * D_theta_cal
					j+=1

		    
			print("For d = " + str(d) + ' MSE = ' + str(round(np.square(y-Y_cal_pred).mean(), 3)))
		
			
			X1_len = len(X1)
			x1 = 0
			while(x1<X1_len):
				cal_nonLine_eqa = 0
				
				for i in range(d+1):

					cal_nonLine_eqa += theta_cal[i] * (math.sin(k * i * X1[x1]) * math.sin(k * i * X1[x1]))

				Y1.append(a0 + cal_nonLine_eqa)

				x1+=1


			# Every 'k' and 'd' value's mean squared error are saved.
			#error_calculate 
			error_calculate[k-1][d] = round(np.square(y-Y_cal_pred).mean(), 3)

			# Sco-efficients in the coefficientsMatrix are saved
			coeffTrigFunct[k-1][d] = [a0] + list(theta_cal)

			k_index.append({'X' : X1, 'Y' : Y1})

			d+=1
       
	   #figure_Plot
		figure_Plot.append(k_index)

		k+=1
		

	# build plots
	for k_index, k_plot in enumerate(figure_Plot):
        
		# calling genPlotGraph 
		genPlotGraph(k_index, k_plot)

	k_best = np.where(error_calculate==np.amin(error_calculate))[0][0]+1
	d_best = np.where(error_calculate==np.amin(error_calculate))[1][0]
	
	print("Least Training Error: ", np.amin(error_calculate))
	print("Least Training Error occurred when 'k': ", str(k_best) + ' d: ' + str(d_best))
	


# Generate plots for K and D
def genPlotGraph(k_index, pItem):

	# using Matplotlib, to create a Sine function plot.
	figu = plt.figure()

	#counting all of the pItems
	for index,item in enumerate(pItem):

        #giving the sine functions their lablles
		plt.plot(item["X"], item["Y"], '-o', label=str(index))
	plt.title("Trigometric Sine Funct for K ->  " + str(k_index+1))
	plt.scatter(X, y, color = '#77c999')
	plt.legend(loc='upper left')
	plt.xlabel('X-axis', fontsize=17,color="green")
	plt.ylabel('Y-axis', fontsize=17,color="green")

	#plt.savefig("Q1AB_K" + str(k_index+1) + ".png")
	plt.show()	
	
def main():
	print('START Q1_AB\n')
	'''
	Start writing your code here
	'''

	# calling Line_Reg
	Line_Reg()
	print('END Q1_AB\n')


inserting_data = './datasets/Q1_B_train.txt'

input_data = readFile(inserting_data)

# Inputs:
#any form of incoming data that can be transformed into an array.
X = np.asarray([item[0] for item in input_data], dtype=float)

# Targets
y = np.asarray([item[1] for item in input_data], dtype=float)

assert X.shape==y.shape, "Shapes of Inputs and Targets are not matching..." 


if __name__ == "__main__":
    main()