# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:10:27 2021

@author: shahf
"""
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np



class CNN: 
    
    def __init__(self,num_filters):
        self.batch_size = 32
        self.filter_size = 3
        self.stride = 1
        self.padding = 0
        self.num_filters = num_filters
     
        
        self.lr = None
        self.filters = None
        self.bias = None
        self.weights = None
        self.bias2 = None

        
    def init_weights(self,image_size,num_labels):

        # kernel weights and bias 
        # init kernel with the range of the number of colors for 8 bpp (256) and scale the number between 0 - 1 
        self.filters = np.random.randint(0,255,[self.filter_size,self.filter_size,self.num_filters]) / 255 
        # init bias between 
        self.bias = np.zeros(self.num_filters) 
       
        
        conv_output_size = self.conv_output_size(image_size, self.filter_size, self.padding, self.stride)
        
        # fully connected weights and bias
        
        self.weights = np.random.random((self.num_filters * conv_output_size ** 2, num_labels)) * 2 - 1 
        self.bias2 = np.zeros(10)
        
        
    def forward (self,batch_x):
        
        # first layer 
        c1 = self.convolution(batch_x, self.filters, self.padding, self.stride) + \
             self.bias[np.newaxis, np.newaxis, np.newaxis, :]
        
        # activition function - ReLu 
        ac1 = self.relu(c1)
        
        # flattening data preparation for fully connected
        flat = ac1.reshape(ac1.shape[0] ,-1)

        fc = self.fully_connected(flat,self.weights,self.bias2)
        
        # activition function - soft max 
        ac2 = self.softmax(fc)
        
        return c1, ac1, flat, fc, ac2
    
    
    def backward(self, batch, y, cache):
         
         '''
         loking for the derivative of loss / c1 to update weights
         using the chain rule 
         loss / c1 =  ( loss / fc ) * (fc / ac1) * (ac1 / c1) 
         
         loss function is cross entropy
         activition function is softmax
         first derivative is loss/ac2 
         equal to y - y.hat
         '''

         # unpack all layers from forward
         c1, ac1, flat, fc, ac2 = cache
                     
         # loss / fc 
         error = ac2 - y
         
         # to find the derivative of fc/ ac1 we need to reshape the flat layer into the original shape 
         # fully connected layer equal to ac1 * w + b  the derivative of fc/ac1 is just w 
         fc_ac1 = error @ self.weights.T
         fc_ac1 = fc_ac1.reshape(ac1.shape)
         
         #the derivative of loss/w is ac1 
         
         loss_w = flat.T @ error
         
         #the derivative of loss/bias is 1 and we sum error for getting 10 biases
         loss_bias = np.sum(error)
     
         # ac1 / c1 the drivative of ReLu is 0 if negative 1 if positive 
         ac1_c1 = np.heaviside(c1, 0)
         
         error2 = fc_ac1 * ac1_c1
         
         error2_conv = self.convolution(batch, error2, self.padding, self.stride).mean(axis=0)
         error2_bias = np.sum( error2 ,axis=(0, 1, 2) )
         
        
         
         self.weights -= self.lr * loss_w
         self.bias2 -= self.lr * loss_bias
         self.filters -= self.lr * error2_conv 
         self.bias -= self.lr * error2_bias
         
         
         
    def fit(self, x, y,lr,epochs = 1000, batch_size=32, epoch_step=10):
        
        n_labels = np.unique(y)
        
        self.init_weights(x.shape[1], n_labels.shape[0])
        self.lr = lr
        
        onehot_y = self.one_hot(y)
        for epoch in range(epochs):
            batches, batches_y = self.minibatch(x, onehot_y)
            
            
            for batch, batch_y in zip(batches, batches_y):
                cache = self.forward(batch)
                self.backward(batch, batch_y, cache)
            if epoch % epoch_step == 0:
                loss_now = self.cross_entropy_loss(self.forward(x)[-1], onehot_y).mean()
                print(f'epoch {epoch} loss {loss_now:.8f}')

    def conv_output_size(self,image_size,filter_size,padding,stride):
        '''
        calculate by the furmula w' = (w-f+2p)/ s + 1 where : 
        w = image size f - filter size p-padding size - s -stride size 
        next_layer_size = w_tag
        '''
        
        w_tag = int((image_size - filter_size + 2 * padding) / stride  + 1)
       
        return w_tag
    def convolution(self, inputs, filters, padding, stride):
        
        # we don't want to pad mini batch so we put (0,0) for zero padding
        input_pad = np.pad(inputs, ((0, 0), (padding, padding), (padding, padding)))
        
        #filters size is HxWxNumber of filteres
        #we assume that that height = weight in filter  (could be filters.shape[0] also )
        f_size = filters.shape[1]
        
        #input_pad is input size + padding size in form of batch size x input+pad x input+pad 
        
        i_size = input_pad.shape[1]
        
        # use this function for foraward and for backward need to add exatra "empty" dimension for mini batch 
        
        f = filters[np.newaxis, :, :, :] if len(filters.shape) == 3 else filters
        
        # using the formula of calculating next layer
        output_size = self.conv_output_size(inputs.shape[1], f_size, padding, stride)
        
        # prepare output layer
        output = np.zeros((inputs.shape[0], output_size, output_size, filters.shape[-1]))
        
        nx = 0
        for idx in range(0, i_size - f_size , stride):
            ny = 0
            for idy in range(0, i_size - f_size , stride):
                
                #we take every time small window and do the convolution on him
                windows = input_pad[:, idx:idx + f_size, idy:idy + f_size]
                output[:, nx, ny, :] = (windows[:, :, :, np.newaxis] * f).sum(axis=2).sum(axis=1)
                ny += 1
            nx += 1
        return output
    
    def normolazie_data():
        #todo
        pass
    
    def minibatch(self,x,y):    
        #number of sample for each batch -  from 32 to number of samples jump by 32
        idx = np.arange(self.batch_size , len(x), self.batch_size)
        # make the batch choise random sample 
        shuffle = np.random.choice(len(x), len(x), replace=False)
        
        # shuffle x,y will split by 32 samples 
        batch_x = np.split(x[shuffle,:,:],idx)
        batch_y = np.split(y[shuffle],idx)
    
        return batch_x,batch_y
    
    def one_hot(self,vec):
        r = np.zeros((vec.shape[0], np.unique(vec).shape[0]))
        r[range(len(vec)), vec] = 1
        return r
    
    def relu(self,x):
    # if X positive x = x else x = 0 
        x[x<0] = 0
        return x
    
    def fully_connected(self,x,w,bias):   
        return x @ w + bias

    @staticmethod
    def softmax(z):    
        e = np.exp(z - z.max(axis=1).reshape(-1, 1))
        return e / e.sum(axis=1).reshape(-1, 1)
    
    @staticmethod
    def cross_entropy_loss(a, y):
        return -np.log((a * y).sum(axis=1))

    def predict(self, x):
        return self.forward(x)[-1].argmax(axis=1)
    



if __name__ == '__main__':

    np.random.seed(42)
    
    

    data = load_digits()
    x = data.images
    y = data.target

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    
    model = CNN(num_filters=6)
    model.fit(x_train, y_train, epochs=200 , lr=1e-3)
    y_pred = model.predict(x_test)
    print(f'accuracy: {accuracy_score(y_test, y_pred):.4f}')

    wrongs_x = x_test[y_pred != y_test, :, :]
    wrongs_y = y_test[y_pred != y_test]
    wrongs_p = y_pred[y_pred != y_test]
