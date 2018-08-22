import numpy as np
from scipy import optimize

class Neural_Network():
	def __init__(self):
		self.inputLayerSize=2
		self.outputLayerSize=1
		self.hiddenLayerSize=3

		self.W1=np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2=np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

	def Forward(self,X):
		self.z2=np.dot(X,self.W1)
		self.a2=self.sigmoid(self.z2)
		self.z3=np.dot(self.a2,self.W2)
		yHat=self.sigmoid(self.z3)

		return yHat

	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	def sigmoidprime(self,z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costfunction(self,X,y):
		self.yHat=self.Forward(X)
		J = 0.5*sum((y-self.yHat)**2)
		return J

	def costFunctionPrime(self,X,y):
		self.yHat=self.Forward(X)

		delta3=np.multiply(-(y-self.yHat),self.sigmoidprime(self.z3))
		dJdW2=np.dot(self.a2.T,delta3)
		delta2=np.dot(delta3,self.W2.T)*self.sigmoidprime(self.z2)
		dJdW1=np.dot(X.T,delta2)

		return dJdW2,dJdW1


	#Helper Functions for interacting with other classes:
	def getParams(self):
        #Get W1 and W2 unrolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
    
	def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class Trainer(object):
	def __init__(self,N):
		self.N=N

	def costfunctionwrapper(self,params,X,y):
		self.N.setParams(params)  #####For Initializing the parameters
		cost=self.N.costfunction(X,y)
		grads=self.N.computeGradients(X,y)

		return cost,grads

	def callbackF(self,params):
		self.N.setParams(params)
		self.J.append(self.N.costfunction(self.X,self.y))

	def Train(self,X,y):
		self.X=X
		self.y=y

		self.J=[]

		params0=self.N.getParams()
		options={'maxiter':200 ,'disp':True}

		_res=optimize.minimize(self.costfunctionwrapper,params0,\
			jac=True,method="BFGS",args=(X,y),\
			options=options,\
			callback=self.callbackF)

		self.N.setParams(_res.x)
		self.optimizationResults=_res

NN=Neural_Network()
T=Trainer(NN)
X=np.array(([3,5],[5,1],[12,2],[6,1.5]),dtype=float)
y=np.array(([35],[51],[12],[61]),dtype=float)
T.Train(X,y)