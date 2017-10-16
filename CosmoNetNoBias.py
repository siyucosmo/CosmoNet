import tensorflow as tf
import numpy as np
from io_Cosmo import *
import hyper_parameters_Cosmo as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def lrelu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def evaluate(val_pred, val_label):
        pred = np.asarray(val_pred)[0]
        val = val_label.eval()
        val_accuracy = np.sqrt(np.sum((pred-val)**2)/(len(val_pred)*2))
        return val_accuracy
        #num_accuracy = tf.reduce_sum(tf.sqrt((val_pred[:,0]-val_label[:,0])**2+(val_pred[:,1]-val_label[:,1])**2))
        #val_accuracy = num_accuracy/tf.cast(val_pred.shape[0],np.float32)

class CosmoNet:
    def __init__(self,train_data,train_label,using_val = False, val_data = None, val_label = None):
        self.train_data = train_data
        self.train_label = train_label
        self.using_val = using_val 
        self.val_data = val_data
        self.val_label = val_label
        
        self.num_parameters = 3*3*3*1*2+4*4*4*2*12+4*4*4*12*64+3*3*3*64*64+2*2*2*64*128+2*2*2*128*12+1024*1024+1024*256+256*2
        
        #initialize weight and bias
        self.W = {}
    
    def deepNet(self,inputBatch,IS_TRAINING,keep_prob):
        # First convolutional layer
        with tf.name_scope('conv1'):
            self.W['W_conv1'] = weight_variable([3, 3, 3, 1, 2])
            h_conv1 = tf.layers.batch_normalization(lrelu(tf.nn.conv3d(inputBatch, self.W['W_conv1'],strides = [1,1,1,1,1],padding = 'VALID'),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
        
        with tf.name_scope('pool1'):
            h_pool1 = tf.nn.avg_pool3d(h_conv1, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
            
        #Second convoluational layer
        with tf.name_scope('conv2'):
            self.W['W_conv2'] = weight_variable([4, 4, 4, 2, 12])
            h_conv2 = tf.layers.batch_normalization(lrelu(tf.nn.conv3d(h_pool1, self.W['W_conv2'],strides = [1,1,1,1,1],padding = 'VALID'),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
            
        with tf.name_scope('pool2'):
            h_pool2 = tf.nn.avg_pool3d(h_conv2, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
        
        #Third convoluational layer
        with tf.name_scope('conv3'):
            self.W['W_conv3'] = weight_variable([4,4,4,12,64])
            h_conv3 = tf.layers.batch_normalization(lrelu(tf.nn.conv3d(h_pool2, self.W['W_conv3'],strides = [1,2,2,2,1],padding = 'VALID'),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
        
        #Fourth convoluational layer
        with tf.name_scope('conv4'):

            self.W['W_conv4'] = weight_variable([3,3,3,64,64])
            h_conv4 = tf.layers.batch_normalization(lrelu(tf.nn.conv3d(h_conv3, self.W['W_conv4'],strides = [1,1,1,1,1],padding = 'VALID'),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
        
        #Fifth convolutional layer
        with tf.name_scope('conv5'):
            self.W['W_conv5'] = weight_variable([2,2,2,64,128])
            h_conv5 = tf.layers.batch_normalization(lrelu(tf.nn.conv3d(h_conv4, self.W['W_conv5'],strides = [1,1,1,1,1],padding = 'VALID'),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
            
        #Sixth convolutional layer
        with tf.name_scope('conv6'):
            self.W['W_conv6'] = weight_variable([2,2,2,128,128])
            h_conv6 = tf.layers.batch_normalization(lrelu(tf.nn.conv3d(h_conv5, self.W['W_conv6'],strides = [1,1,1,1,1],padding = 'VALID'),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
        
        with tf.name_scope('fc1'):
            h_conv6_flat = tf.reshape(h_conv6,[-1,1024])
            self.W['W_fc1'] = weight_variable([1024,1024])
            h_fc1 = tf.layers.batch_normalization(lrelu(tf.matmul(h_conv6_flat, self.W['W_fc1']),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
        
        with tf.name_scope('dropout1'):
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)       
        
        with tf.name_scope('fc2'):
            self.W['W_fc2'] = weight_variable([1024,256])
            h_fc2 = tf.layers.batch_normalization(lrelu(tf.matmul(h_fc1_drop, self.W['W_fc2']),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
            
        with tf.name_scope('dropout2'):
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
            
        with tf.name_scope('fc2'):
            self.W['W_fc3'] = weight_variable([256,2])
            h_fc3 = tf.layers.batch_normalization(lrelu(tf.matmul(h_fc2_drop,self.W['W_fc3']),hp.Model['LEAK_PARAMETER']),training = IS_TRAINING)
            return h_fc3
    
            
    def loss(self,inputBatch,inputLabel,IS_TRAINING,keep_prob):
        with tf.name_scope('loss'):
            predictions = self.deepNet(inputBatch = inputBatch,IS_TRAINING = IS_TRAINING,keep_prob = keep_prob)
            lossL1 = tf.reduce_mean(tf.losses.absolute_difference(labels = inputLabel,predictions = predictions))
            for w in self.W:
                lossL1 += hp.Model["REG_RATE"]*tf.nn.l2_loss(self.W[w])/self.num_parameters
            return lossL1, predictions
    

    def optimize(self,inputBatch,inputLabel, IS_TRAINING,keep_prob):
        loss,predictions = self.loss(inputBatch = inputBatch,inputLabel = inputLabel, IS_TRAINING = IS_TRAINING,keep_prob = keep_prob)
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(hp.Model['LEARNING_RATE']).minimize(loss)
            return train_step, loss, predictions
    
    def train(self):
        keep_prob = tf.placeholder(tf.float32)
        IS_TRAINING = tf.placeholder(tf.bool)
        inputBatch = tf.placeholder(tf.float32,[None,64,64,64,1])
        inputLabel = tf.placeholder(tf.float32,[None,2])
        train_step, loss,predictions = self.optimize(inputBatch =  inputBatch, inputLabel= inputLabel, IS_TRAINING=IS_TRAINING, keep_prob = keep_prob)
        """
        if(self.using_val):
            val_pred = self.deepNet(inputBatch = self.val_data, IS_TRAINING = False,keep_prob = 1)
            num_accuracy = tf.reduce_sum(tf.sqrt((val_pred[:,0]-self.val_label[:,0])**2+(val_pred[:,1]-self.val_label[:,1])**2))
            val_accuracy = num_accuracy/tf.cast((self.val_label).shape[0],np.float32)
            #val_loss = tf.reduce_mean(tf.losses.absolute_difference(labels = self.val_label,predictions = val_pred))
            #for w in self.W:
            #    val_loss += hp.Model["REG_RATE"]*tf.nn.l2_loss(self.W[w])/self.num_parameters
        """
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session() as sess:
            losses = []  
            val_accuracys = []          
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            """
            if(self.using_val):
                val_pred = self.deepNet(self.val_data)
                #val_label_value = tf.cast(self.val_label,tf.float32)
                num_accuracy = tf.reduce_sum(tf.cast(tf.logical_and(tf.abs(val_pred[:,0]-self.val_label[:,0])<0.001, tf.abs(val_pred[:,1]-self.val_label[:,1])<0.001),np.float32))
                val_accuracy = num_accuracy/tf.cast((self.val_label).shape[0],np.float32)
            """
            for i in range(8000):   
                _,loss_= sess.run([train_step,loss],feed_dict = {inputBatch: (self.train_data).eval(), inputLabel: (self.train_label).eval(), IS_TRAINING:True,keep_prob:hp.Model['DROP_OUT']})
                losses.append(loss_)
                if i%50==0:
                    if self.using_val:
                        print (i,loss_)
                        val_pred = sess.run([predictions],feed_dict = {inputBatch: (self.val_data).eval(), inputLabel: (self.val_label).eval(), IS_TRAINING:False,keep_prob:1})
                    	val_accuracy_ = evaluate(val_pred,self.val_label)
                        #val_accuracy_,num_accuracy_,pred_label,true_label= sess.run([val_accuracy,num_accuracy,val_pred[:,0][0],self.val_label[:,0][0]])
                        val_accuracys.append(val_accuracy_)
                        print ("accuracy = " +str(val_accuracy_))
                    	#val_losses = sess.run(val_loss)
                        #print val_losses
            np.savetxt("pred.txt",np.asarray(val_pred)[0])
            np.savetxt("label.txt",(self.val_label).eval()) 
            coord.request_stop();
            coord.join(threads);
        return losses,val_accuracys

if __name__ == "__main__":
    #NbodySimuBatch = tf.placeholder(tf.float32,shape =[None,64,64,64,1])
    #NbodySimuLabel = tf.placeholder(tf.float32,shape =[None,2])
    #trainCosmo = CosmoNet(train_data = NbodySimuBatch, train_label = NbodySimuLabel)
    #losses = trainCosmo.train()
    NbodySimuDataBatch64, NbodySimuLabelBatch64 = readDataSet(filenames = [str(i)+'.tfrecord' for i in range(0,450)])
    NbodySimuDataBatch32, NbodySimuLabelBatch32 = tf.cast(NbodySimuDataBatch64,tf.float32),tf.cast(NbodySimuLabelBatch64,tf.float32)
    valDataBatch64, valLabelbatch64 = readDataSet(filenames=[str(i)+".tfrecord" for i in range(450,499)]);
    valDataBatch32, valLabelbatch32 = tf.cast(valDataBatch64,tf.float32),tf.cast(valLabelbatch64,tf.float32)
    #valDataBatch32, valLabelbatch32 = NbodySimuDataBatch32, NbodySimuLabelBatch32
    trainCosmo = CosmoNet(train_data = NbodySimuDataBatch32, train_label = NbodySimuLabelBatch32, using_val = True, val_data = valDataBatch32, val_label = valLabelbatch32)
    losses,val_accuracys = trainCosmo.train()
    np.savetxt("losses.txt",losses)
    np.savetxt("accuracy.txt",val_accuracys)

    
    
            
