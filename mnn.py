import numpy as np
import math
# The model that will be used for prediction is called a neural network
class NeuralNetwork():
    # This constructor creates dicts that will store the network's weights per layer, error per layer and change in 
    # error per layer. The constructor also initializes the
    def __init__(self):
        self.weights = {}
        self.err = {}
        self.weight_delta = {}
        self.num_layers = 1

    def new_layer(self,shape):
        self.weights[self.num_layers] = 2 * np.random.random(shape) - 1
        #self.err[self.num_layers] = np.zeros(shape)
       # self.weight_delta[self.num_layers] = np.zeros(shape)
        self.num_layers += 1

    def nonlin(self,x,deriv=False):
        if deriv == True:
            return x*(1 - x)
        return 1/(1+np.exp(-x))
    
    def predict(self,data):
        for layer in range(1,self.num_layers):
            data = self.nonlin(np.dot(data,self.weights[layer]))
        return data

    def fit(self,input_data,output_data, epochs):
        #data = input_data
        break_loop = False
        for e in range(epochs):
            data = input_data
            for layer in range(1,self.num_layers):
                print ("<----> Complete Epochs ",e+1,"/",epochs)
                data = self.nonlin(np.dot(data,self.weights[layer]))
                print ("<=--> Complete Epochs ",e+1,"/",epochs)
                # Checks if array still contains numbers
                check_nan = output_data - data
                for array in check_nan:
                    for i in array:
                        if math.isnan(i) == True:
                            break_loop = True
                if break_loop == True:
                    break
                self.err[layer] = output_data - data
                print ("<==--> Complete Epochs ",e+1,"/",epochs)
                self.weight_delta[layer] = np.dot(self.err[layer],self.nonlin(self.weights[layer].T,deriv=True))
                print ("<===-> Complete Epochs ",e+1,"/",epochs)
                self.weights[layer] += data.T.dot(self.weight_delta[layer]).T
                print ("<===> Complete Epochs ",e+1,"/",epochs)
            print("Err : ",model.err)
model = NeuralNetwork()
model.new_layer((3,5))
model.new_layer((5,4))
model.new_layer((4,6))
model.new_layer((6,1))
print("Weights : ",model.weights[1],"\nError : ",model.err,"\nAdjustments : ",model.weight_delta)

x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])
print("Input : ",x)
print("Output : ",y)
print("Weight X Input : ",np.dot(x,model.weights[1]))
print(range(model.num_layers))
model.fit(x,y,300)
print("Err : ",model.err)
print("Input : ",x)
print("Output : ",y)
print(model.weights)
print("Predited Output : ",model.predict(x))

