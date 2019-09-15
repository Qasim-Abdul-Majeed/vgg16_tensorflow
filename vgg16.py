 
import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
from datetime import datetime
from sklearn.metrics import accuracy_score


#Mount a drive
from google.colab import drive
drive.mount('/content/drive')



batch_size = 16
x = tf.placeholder(tf.float32, shape = [batch_size, 224, 224, 3])
y = tf.placeholder(tf.int32, shape = [batch_size])

#Load all weights from a pre-trained model.
data_dict = np.load('/content/drive/My Drive/Months_DataSet/Model_weights/vgg16.npz',
                    allow_pickle = True,
                    encoding='latin1').item()
print('Weights has been loaded!!!')




#Functions definition.

def freeze_filt(name):
    return tf.constant(data_dict[name][0], name = 'f_' + name)

def freeze_bias (name):
    return tf.constant(data_dict[name][1], name = 'b_' + name)

def freeze_conv_layer(inputs, name):
    #Get a filter weights from a pre-trained Model.
    filt = freeze_filt(name)
    
    #Apply a convolution on an input.
    conv = tf.nn.conv2d(inputs,
                        filt, 
                        strides = [1, 1, 1, 1],
                        padding = 'SAME', 
                        name = name)
    
    #Get a bias weights from a pre-trained model.
    bias_val = freeze_bias(name)
    #Add bias.
    bias = tf.nn.bias_add(conv, bias_val)
    
    #Apply an activation function.
    relu = tf.nn.relu(bias)
    
    return relu

def trainable_filt(name):
    return tf.Variable(data_dict[name][0], name = 'f_' + name)

def trainable_bias(name):
    return tf.Variable(data_dict[name][1], name = 'b_' + name)



def trainable_conv_layer(inputs, name):
    #Get a filter weights from a pre-trained model.
    filt = trainable_filt(name)
    
    #Apply a convolution on an input.
    conv = tf.nn.conv2d(inputs,
                        filt,
                        strides = [1, 1, 1, 1],
                        padding = 'SAME',
                        name = name)
    
    #Get a bias weights from a pre-trained model.
    bias_val =  trainable_bias(name)
    
    #Add Bias...
    bias = tf.nn.bias_add(conv, bias_val)
    
    #Apply an activation function
    return tf.nn.relu(bias)
    
    
    
    
#FULLY Connected Layers...

def fc_weights(name, trainable = True):
    if trainable:
        return tf.Variable(data_dict[name][0], name = 'w_' + name)
    else:
        return tf.constant(data_dict[name][0], name = 'w_' + name)

def fc_bias(name, trainable = True):
    if trainable:
        return tf.Variable(data_dict[name][1], name = 'b_' + name)
    else:
        return tf.constant(data_dict[name][0], name = 'b_' + name)
    
            
    

def fc_layer(inputs, name, trainable =  True):
    #Reshape or flatten an inputs... 
    dims = np.prod(inputs.shape[1:]);
    data = tf.reshape(inputs, [-1, dims])
    
    filt = fc_weights(name, trainable)
    fc = tf.matmul(data, filt)
    
    #Load bias weights.
    bias_val = fc_bias(name, trainable)
    
    #Add a bias.
    return tf.nn.bias_add(fc, bias_val)
    
    


def get_variable(name, shape, trainable = False):
    var = tf.get_variable(name,
                          shape,
                          dtype = tf.float32,
                          trainable = True
                         )
    return var



def fc_layer_final(inputs, name = None, ch_out = 1, trainable = True):
    dims = np.prod(inputs.shape[1:]);

    #data = tf.reshape(inputs, [-1, dims])
    
    #filt = fc_weights(name, trainable)
    #fc = tf.matmul(data, filt)

    #Load bias weights.
    #bias_val = fc_bias(name, trainable)

    weigh = get_variable(name = 'f_' + name, shape = [dims, ch_out], trainable = True)
    bias = get_variable(name = 'b_' + name, shape = ch_out, trainable = True);
    flat = tf.reshape(inputs, [-1, dims])
    fc = tf.matmul(flat, weigh);
    return tf.nn.bias_add(fc, bias);




'''

#Read an images...
folder =  "/content/drive/My Drive/Months_DataSet/copy"
images = []
labels = []
count = 0;
for month in tqdm(os.listdir(folder)):
    mon_path = os.path.join(folder, month)
    for img in os.listdir(mon_path):
        temp = cv2.imread(os.path.join(mon_path, img))
        temp.resize(224, 224, 3)
        images.append(temp) #Read an image.
        labels.append(int(month))
        
data = list(zip(images, labels))
random.shuffle(data) #Shuffle data.

images, labels = zip(*data)
'''
#Load images from a npy file.
images = np.load('/content/drive/My Drive/Months_DataSet/copy' + '/images.npy', allow_pickle = True)
labels = np.load('/content/drive/My Drive/Months_DataSet/copy' + '/labels.npy', allow_pickle = True)
print("Total images: ", np.array(images).shape)
print("Labels: ", np.array(labels).shape)


#Start defining a model.
#Layers with pretrained weights values.

#Layers in this cell are freezed.

conv1_1 = freeze_conv_layer(x, 'conv1_1')
conv1_2 = freeze_conv_layer(conv1_1, 'conv1_2')
pool_1 = tf.nn.max_pool(conv1_2,
                       ksize = [1, 2, 2, 1],
                       strides = [1, 2, 2, 1],
                       padding = 'SAME',
                       name = 'max_pool_1')

conv2_1 = freeze_conv_layer(pool_1, 'conv2_1')
conv2_2 = freeze_conv_layer(conv2_1, 'conv2_2')
pool_2 = tf.nn.max_pool(conv2_2,
                        ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME',
                        name = 'max_pool_2')

conv3_1 = freeze_conv_layer(pool_2, 'conv3_1')
conv3_2 = freeze_conv_layer(conv3_1, 'conv3_2')
conv3_3 = freeze_conv_layer(conv3_2, 'conv3_3')
pool_3 = tf.nn.max_pool(conv3_3,
                        ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME',
                        name = 'max_pool_3')

conv4_1 = freeze_conv_layer(pool_3, 'conv4_1')
conv4_2 = freeze_conv_layer(conv4_1, 'conv4_2')
conv4_3 = freeze_conv_layer(conv4_2, 'conv4_3')
pool_4 = tf.nn.max_pool(conv4_3,
                        ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME',
                        name = 'max_pool_4')


#Trainable layers Onwards...

conv5_1 = trainable_conv_layer(pool_4, 'conv5_1')
conv5_2 = trainable_conv_layer(conv5_1, 'conv5_2')
conv5_3 = trainable_conv_layer(conv5_2, 'conv5_3')
pool_5 = tf.nn.max_pool(conv5_3, 
                        ksize = [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME',
                        name = 'max_pool_5')

#Fully Connected Layers...

fc_6 = fc_layer(pool_5, 'fc6', trainable = True)
relu_6 = tf.nn.relu(fc_6)

fc_7 = fc_layer(relu_6, 'fc7', trainable = True)
relu_7 = tf.nn.relu(fc_7)

#Here is a final layer for our months classification.

fc_8 = fc_layer_final(relu_7, name = 'fc8', ch_out = 12);


#Calculate probabilities for each month.
prob = tf.nn.softmax(fc_8, name = 'prob');



#Define optimizer and its loss...
#labels = tf.convert_to_tensor(y[:80], dtype = tf.int32);
cost = tf.losses.sparse_softmax_cross_entropy(labels = y,
                                              logits = prob#)
                                              #weights=1.0,\
                                              #scope=None,
                                              #loss_collection=tf.GraphKeys.LOSSES,
                                              #reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
                                              )




#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)
#Run an optimizer with a tf.placeholder...
saver = tf.train.Saver() #Trainer to save a model.
    
sess = tf.Session()
sess.run(tf.global_variables_initializer())
epoch_num = 1
T_images = np.array(labels).shape[0] #Total number of images.
T_train_images = int(T_images * 0.8)
T_val_images = int(T_images * 0.2)
while epoch_num < 20:
  count = 0;
  while count < T_train_images:
      sess.run(optimizer, feed_dict = ({x: images[count:count+batch_size], y: labels[count:count+batch_size]}))
      count +=batch_size
      
  print('Epoch ', epoch_num, ' is done Sucessfully done.')
  #Saving weights or ckpt file of a trained model.
  ckpt_path = '/content/drive/My Drive/Months_DataSet/Model_weights_months'
  saver.save(sess, os.path.join(ckpt_path, str(epoch_num) + '_' + str(datetime.now()) + 'mnist.ckpt'))
  print('ckpt saved path is: ', ckpt_path)

  #print('Calculating Validation accuracy')
  val_prob = []
  while count < T_images:
    temp = sess.run(prob, feed_dict = {x: images[count: count + batch_size]})
    val_prob.append(temp)
    count +=batch_size

  #Now get an element with a highest probability.
  val_labels = []
  
  for i in range(int(T_val_images/batch_size)):
    for j in range(batch_size):
      val_labels.append(np.argmax(val_prob[i][j]))

  #Now calculating accuracy.
  acc = accuracy_score(labels[T_train_images:], val_labels)
  print('Test Accuracy for epoch ', epoch_num, ' is: ' , acc)
  epoch_num  +=1

print('Sucessfully done.')                                 
                          