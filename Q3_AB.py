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
	


def normalize(x):

    np.random.seed(123)
    x_norm = x / np.linalg.norm(x)    
    return x_norm


def logisticRegression():
	iterations = 20

	# learning rate
	lr = 0.01

	# Batch size
	batch_size = 3

	# Number of samples in the training set
	m = X.shape[0]

	# Intialize Weights and Biases 
	W = np.zeros((X.shape[1], 1)) 
	b = 0
	losses = []
	gender_probas = []


	for itera in range(iterations):
		for i in range(0, m, batch_size):
			# Creating batches for gradient descent
			X_batch = X[i:i+batch_size, :]
			y_batch = y[i:i+batch_size, :]

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
		losses.append(loss)
		#print("Loss: ", loss)
	
		# Predicting the class label
		count = 0
		for i in range(m):
			x = X[i:i+1, :]
			z = np.dot(x, W) + b
			y_prob = 1.0 / (1 + np.exp(-z))
			y_pred = 1.0 if y_prob > 0.5 else 0.0
			gender_probas.append(y_prob)
		
			if y_pred==y[i][0]:
				count += 1

		print("Iteration :",itera, " Accuracy in prediction:", round((count/m) * 100, 2))
		#print("Number of samples: ", m)
	
	
	plt.plot(list(range(1,iterations+1)), losses)
	plt.title('Logistic Loss function')
	plt.xlabel('Iterations', color='#1C2833')
	plt.ylabel('Loss', color='#1C2833')
	plt.legend(loc='upper left')
	plt.grid()
	#plt.savefig("Q3AB_LogisticLoss.pdf")
	plt.show()

	x_ = np.linspace(min(X[:,0]),max(X[:,0]),100)
	y_ = np.linspace(min(X[:,1]),max(X[:,1]),100)

	X_,Y_ = np.meshgrid(x_,y_)
	Z_ = (b - W[0,0]*X_ - W[1,0]*Y_) / W[2,0]


	#Ploting the 3d graph-2
	fig = plt.figure(figsize = (6,6))
	ax = plt.axes(projection='3d')
	ax.set_xlabel('height')
	ax.set_ylabel('weight')
	ax.set_zlabel('age')

	# x = X[i:i+1, :]
	z = np.dot(X, W) + b
	y_prob = 1.0 / (1 + np.exp(-z))
	y_pred = [ 1 if val[0] > 0.5 else 0 for val in y_prob]

	X_F = np.array([ X[ind] for ind in range(y.shape[0]) if y_pred[ind]==1])
	X_M = np.array([ X[ind] for ind in range(y.shape[0]) if y_pred[ind]==0])
	ax.scatter3D(X_M[:,0], X_M[:,1], X_M[:,2] )
	ax.scatter3D(X_F[:,0], X_F[:,1], X_F[:,2], color = "red" )
	ax.plot_surface(X_, Y_, Z_)
	ax.azim = 10
	ax.elev = 10
	plt.show()
	

loading_file = './datasets/Q3_data.txt'
data = readFile(loading_file)
# Inputs:
X = np.asarray([item[:-1] for item in data], dtype=float)
# Targets [W - 1, M - 0]
y = np.asarray([1 if item[-1]=='W' else 0 for item in data], dtype=float)
y = y.reshape(y.shape[0],1)
assert X.shape[0]==y.shape[0], "Shapes of Inputs and Targets are not matching..." 


def main():
	print('START Q3_AB\n')
	'''
	Start writing your code here
	'''
	logisticRegression()
	print('END Q3_AB\n')


if __name__ == "__main__":
    main()
