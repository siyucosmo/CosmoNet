## for this one, change the order between relu and batch

import tensorflow as tf
import numpy as np
from io_Cosmo import *
import hyper_parameters_Cosmo as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_aug import *
import time
from numpy import linalg as LA

#def weight_variable(shape):
#        initial = tf.truncated_normal(shape, stddev=0.1)
#        return tf.Variable(initial)

test_random = 0

def weight_variable(shape,name):
	W = tf.get_variable(name,shape=shape, initializer=tf.contrib.layers.xavier_initializer())
	return W

def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def lrelu(x, alpha):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class CosmoNet:
    def __init__(self,train_data,train_label,using_val = False, val_data = None, val_label = None):
        self.train_data = train_data
        self.train_label = train_label
        self.using_val = using_val 
        self.val_data = val_data
        self.val_label = val_label
        
        #self.num_parameters = 3*3*3*1*2+4*4*4*2*12+4*4*4*12*64+3*3*3*64*64+2*2*2*64*128+2*2*2*128*12+1024*1024+1024*256+256*2
        self.num_parameters = 1

        #initialize weight and bias
        self.W = {}
        self.b = {}
        self.bn_param = {}
        self.W['W_conv1'] = weight_variable([3, 3, 3, 1, 2],'w1')
	self.b['b_conv1'] = bias_variable([2])
	self.W['W_conv2'] = weight_variable([4, 4, 4, 2, 12],'w2')
	self.b['b_conv2'] = bias_variable([12])
	self.W['W_conv3'] = weight_variable([4,4,4,12,64],'w3')
	self.b['b_conv3'] = bias_variable([64])
	self.W['W_conv4'] = weight_variable([3,3,3,64,64],'w4')
	self.b['b_conv4'] = bias_variable([64])
        self.W['W_conv5'] = weight_variable([2,2,2,64,128],'w5')
        self.b['b_conv5'] = bias_variable([128])
	self.W['W_conv6'] = weight_variable([2,2,2,128,128],'w6')
	self.b['b_conv6'] = bias_variable([128])
	self.W['W_fc1'] = weight_variable([1024,1024],'w7')
        self.b['b_fc1'] = bias_variable([1024])
	self.W['W_fc2'] = weight_variable([1024,256],'w8')
        self.b['b_fc2'] = bias_variable([256])
	self.W['W_fc3'] = weight_variable([256,2],'w9')
        self.b['b_fc3'] = bias_variable([2])

    #Define some fuctions that might be used   
    
    def BatchNorm(self,inputT, IS_TRAINING, scope,reuse=None):
	with tf.variable_scope(scope,'model',reuse = reuse):
	    return tf.contrib.layers.batch_norm(inputT, is_training=IS_TRAINING,center = True, scale = True,epsilon=0.0001,decay=0.99,scope=scope)
   	    #tf.layers.batch_normalization(inputT,training=training,epsilon=0.0001,axis=-1,name=scope)

    def deepNet(self,inputBatch,IS_TRAINING,keep_prob,scope,reuse):
        # First convolutional layer
        with tf.name_scope('conv1'):
            h_conv1 = lrelu(self.BatchNorm(tf.nn.conv3d(inputBatch,self.W['W_conv1'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv1'],IS_TRAINING = IS_TRAINING, scope = scope+str(1), reuse = reuse),hp.Model['LEAK_PARAMETER'])
        
	with tf.name_scope('pool1'):
            h_pool1 = tf.nn.avg_pool3d(h_conv1, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
            
        #Second convoluational layer
        with tf.name_scope('conv2'):
            h_conv2 = lrelu(self.BatchNorm(tf.nn.conv3d(h_pool1, self.W['W_conv2'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv2'],IS_TRAINING=IS_TRAINING,scope = scope+str(2),reuse = reuse),hp.Model['LEAK_PARAMETER'])
            
        with tf.name_scope('pool2'):
            h_pool2 = tf.nn.avg_pool3d(h_conv2, ksize=[1,2,2,2,1], strides = [1,2,2,2,1], padding = 'VALID')
        
        #Third convoluational layer
        with tf.name_scope('conv3'):
            h_conv3 = lrelu(self.BatchNorm(tf.nn.conv3d(h_pool2, self.W['W_conv3'],strides = [1,2,2,2,1],padding = 'VALID') + self.b['b_conv3'],IS_TRAINING=IS_TRAINING, scope = scope+str(3),reuse=reuse),hp.Model['LEAK_PARAMETER'])
        
        #Fourth convoluational layer
        with tf.name_scope('conv4'):
            h_conv4 = lrelu(self.BatchNorm(tf.nn.conv3d(h_conv3, self.W['W_conv4'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv4'],IS_TRAINING=IS_TRAINING,scope = scope+str(4),reuse=reuse),hp.Model['LEAK_PARAMETER'])
        
        #Fifth convolutional layer
        with tf.name_scope('conv5'):
            h_conv5 = lrelu(self.BatchNorm(tf.nn.conv3d(h_conv4, self.W['W_conv5'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv5'],IS_TRAINING=IS_TRAINING,scope = scope+str(5),reuse=reuse),hp.Model['LEAK_PARAMETER'])
            
        #Sixth convolutional layer
        with tf.name_scope('conv6'):
            h_conv6 = lrelu(self.BatchNorm(tf.nn.conv3d(h_conv5, self.W['W_conv6'],strides = [1,1,1,1,1],padding = 'VALID') + self.b['b_conv6'],IS_TRAINING=IS_TRAINING,scope = scope+str(6),reuse=reuse),hp.Model['LEAK_PARAMETER'])
        
        with tf.name_scope('fc1'):
            h_conv6_flat = tf.reshape(h_conv6,[-1,1024])
            h_fc1 = lrelu(tf.matmul(tf.nn.dropout(h_conv6_flat,keep_prob), self.W['W_fc1']) + self.b['b_fc1'],hp.Model['LEAK_PARAMETER'])
        
        with tf.name_scope('fc2'):
            h_fc2 = lrelu(tf.matmul(tf.nn.dropout(h_fc1,keep_prob), self.W['W_fc2']) + self.b['b_fc2'],hp.Model['LEAK_PARAMETER'])
            
        with tf.name_scope('fc3'):
            h_fc3 = tf.matmul(tf.nn.dropout(h_fc2,keep_prob), self.W['W_fc3']) + self.b['b_fc3']
            return h_fc3
    
            
    def loss(self):
        with tf.name_scope('loss'):
            predictions = self.deepNet(inputBatch = self.train_data,IS_TRAINING = True,keep_prob = hp.Model['DROP_OUT'],scope='conv_bn',reuse = None)
            lossL1 = tf.reduce_mean(tf.abs(self.train_label-predictions))
            for w in self.W:
                lossL1 += hp.Model["REG_RATE"]*tf.nn.l2_loss(self.W[w])/self.num_parameters
            return lossL1
    
    def validation_loss(self):
        val_predict = self.deepNet(inputBatch = self.val_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        val_predict = val_predict*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        val_true = self.val_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Val = tf.reduce_mean(tf.abs(val_true-val_predict)/val_true)
        return lossL1Val,val_true,val_predict

    def train_loss(self):
	train_predict = self.deepNet(inputBatch = self.train_data,IS_TRAINING = False,keep_prob = 1,scope='conv_bn',reuse=True)
        train_predict = train_predict*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        train_true = self.train_label*tf.constant([2.905168635566176411e-02,4.023372385668218254e-02],dtype = tf.float32)+tf.constant([2.995679839999998983e-01,8.610806619999996636e-01],dtype = tf.float32)
        lossL1Train = tf.reduce_mean(tf.abs(train_true-train_predict)/train_true)
	return lossL1Train,train_true,train_predict

    def optimize(self):
        loss = self.loss()
        with tf.name_scope('adam_optimizer'):
	    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	    with tf.control_dependencies(update_ops):
	        train_step = tf.train.AdamOptimizer(hp.Model['LEARNING_RATE']).minimize(loss)

	lossL1Train,train_true,train_predict = self.train_loss()    
        return train_step, loss,lossL1Train,train_true,train_predict
    
    def train(self):
        
        train_step, loss, lossL1Train,train_true,train_predict = self.optimize()
        lossL1Val,val_true,val_predict = self.validation_loss()
        
	config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session() as sess:
            losses_train = []  
            losses_val = []
            losses = []
	    val_accuracys = []       
	    data_accuracys = []   
            sess.run(tf.global_variables_initializer())
	    sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
	    for epoch in range(hp.RUNPARAM['num_epoch']):
		start_time = time.time()
                loss_per_epoch_train = 0
                loss_per_epoch_val = 0
                lossTrain_per_epoch = 0
                for i in range(hp.RUNPARAM['batch_per_epoch']):  
			_,lossTrain,lossL1Train_,train_true_,train_predict_ = sess.run([train_step,loss,lossL1Train,train_true,train_predict])
			loss_per_epoch_train += lossL1Train_
                        lossTrain_per_epoch +=lossTrain
                losses.append(lossTrain_per_epoch/hp.RUNPARAM['batch_per_epoch'])
		losses_train.append(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
		
		for i in range(hp.RUNPARAM['batch_per_epoch_val']):
			loss_,val_true_,val_predict_ = sess.run([lossL1Val,val_true,val_predict])
                        loss_per_epoch_val += loss_
		losses_val.append(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'])
                print("Epoch {} took {:.3f}s".format(epoch, time.time() - start_time))
		print "  training loss: %.3f" %(loss_per_epoch_train/hp.RUNPARAM['batch_per_epoch'])
		print "  validation loss: %.3f" %(loss_per_epoch_val/hp.RUNPARAM['batch_per_epoch_val'])
                np.savetxt('loss_train_batch20_4.txt',losses_train)
                np.savetxt('loss_val_batch20_4.txt',losses_val)
                np.savetxt('losses4.txt',losses)
                np.savetxt('/zfsauton/home/siyuh/pred4/train_pred'+str(epoch)+'.txt',np.c_[train_true_,train_predict_])
                np.savetxt('/zfsauton/home/siyuh/pred4/val_pred'+str(epoch)+'.txt',np.c_[val_true_,val_predict_])
		
                        
            
            coord.request_stop();
            coord.join(threads);
        return losses_train,losses_val

if __name__ == "__main__":
    NbodySimuDataBatch64, NbodySimuLabelBatch64 = readDataSet(filenames = [str(i)+'.tfrecord' for i in range(0,450)])
    NbodySimuDataBatch32, NbodySimuLabelBatch32 = tf.cast(NbodySimuDataBatch64,tf.float32),tf.cast(NbodySimuLabelBatch64,tf.float32)
    valDataBatch64, valLabelbatch64 = readDataSet(filenames=[str(i)+".tfrecord" for i in range(450,495)]);
    valDataBatch32, valLabelbatch32 = tf.cast(valDataBatch64,tf.float32),tf.cast(valLabelbatch64,tf.float32)
    trainCosmo = CosmoNet(train_data = NbodySimuDataBatch32, train_label = NbodySimuLabelBatch32, using_val = True, val_data = valDataBatch32, val_label = valLabelbatch32)
    losses_train,losses_val = trainCosmo.train()
    #np.savetxt("losses4.txt",losses)
    #np.savetxt("accuracy4.txt",val_accuracys)
    #np.savetxt("data_accuracy4.txt",data_accuracys)

    
    
            
