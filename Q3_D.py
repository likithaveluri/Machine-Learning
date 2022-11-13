import numpy as np
import matplotlib.pyplot as plt

def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    input_np = np.array(input_data)
    return input_np


def logisticRegressionLOOV():
	iterations = 20

	# learning rate
	lr = 0.01

	# Batch size
	batch_size = 3

	#Samples in the training set
	m = X.shape[0]

	# Intialize Weights and Biases 
	W = np.zeros((X.shape[1], 1)) 
	b = 0
	losses = []
	gender_probas = []
	accuracies = []

	j = 0

	while(j < X.shape[0]):
		X_mod = np.delete(X, (j), axis=0)
		y_mod = np.delete(y, (j), axis=0)

		# Intialize Weights and Biases 
		W = np.zeros((X.shape[1], 1))
		b = 0
		
		gender_probas = []

		for ite in range(iterations):
			for i in range(0, X_mod.shape[0], batch_size):
				# Creating batches for gradient descent
				X_batch = X_mod[i:i+batch_size, :]
				y_batch = y_mod[i:i+batch_size, :]

				# Applying Sigmoid function (1 / (1 + exp(-z))) and gradient descent
				z = np.dot(X_batch, W) + b
				y_prob = 1.0/(1 + np.exp(-z))
				diff_W = -(1/m)*np.dot(X_batch.T, (y_prob - y_batch))
				diff_b = -(1/m)*np.sum((y_prob - y_batch))

				# Updating Weights and biases
				W += lr * diff_W
				b += lr * diff_b
		
			# Evaluating losses with the updated weights and biases
			z = np.dot(X, W) + b
			y_prob = 1.0 / (1 + np.exp(-z))
			loss = -np.mean(y*(np.log(y_prob)) - (1-y)*np.log(1-y_prob))
			#losses.append(loss)
			#print("Loss: ", loss)
		
		# Append loss obtained in each iteration
		losses.append(loss)

		# Predicting the class label
		count = 0
		for i in range(1):
			x = X[j, :]
			z = np.dot(x, W) + b
			y_prob = 1.0 / (1 + np.exp(-z))
			y_pred = 1.0 if y_prob > 0.5 else 0.0
			if y_pred==y[j][0]:
				count += 1
		
		#print("Accuracy when index " + str(j) + " removed:", round((count/m)*100, 2))
		accuracies.append(round((count), 2))

		j += 1

	print("Height, Weight")
	print("For Alpha : ", lr, " iteration : ", iterations)
	print("Leave-one-out Accuracy : ", round((sum(accuracies)/m)*100,2))
	
	# X_ = np.linspace(min(X[:,0]),max(X[:,0]),100)
	# Y_ = (b - W[0,0]*X_ ) / W[1,0]

	# #Ploting the 3d graph-2
	# fig = plt.figure(figsize = (6,6))
	# ax = plt.axes()
	# ax.set_xlabel('height')
	# ax.set_ylabel('weight')

	# z = np.dot(X, W) + b
	# y_prob = 1.0 / (1 + np.exp(-z))
	# y_pred = [ 1 if val[0] > 0.55 else 0 for val in y_prob]

	# X_F = np.array([ X[ind] for ind in range(y.shape[0]) if y_pred[ind]==1])
	# X_M = np.array([ X[ind] for ind in range(y.shape[0]) if y_pred[ind]==0])
	# print(X_M.shape)
	# ax.scatter(X_M[:,0], X_M[:,1], color = "red" )
	# ax.scatter(X_F[:,0], X_F[:,1], color = "red" )
	# ax.scatter(X_, Y_)
	# plt.plot()


loading_file = './datasets/Q3_data.txt'
data = readFile(loading_file)
# Inputs:
X = np.asarray([item[:-1] for item in data], dtype=float)
# Removing age feature
X = X[:,:-1]
# Targets [W - 1, M - 0]
y = np.asarray([1 if item[-1]=='W' else 0 for item in data], dtype=float)
y = y.reshape(y.shape[0],1)
assert X.shape[0]==y.shape[0], "Shapes of Inputs and Targets are not matching..." 


def main():
	print('START Q3_D\n')
	'''
	Start writing your code here
	'''
	logisticRegressionLOOV()
	print('END Q3_D\n')


if __name__ == "__main__":
    main()
    
"""
In Naive bayies and knn , removing of age data does not impact the model performace
The accuracy of the model increases when removed the age feature, when compared to having it, because the features become separte and this gives the better accuracy
KNN and naive bayes work better than logistic because they are probability approches.
"""