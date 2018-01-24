import mxnet as mx
import numpy as np
from PIL import Image
import logging
import numpy as np

def resize_batchimage(data, target_dimension):
    shape_size = data.shape
    big_array = np.zeros((shape_size[0], 3, target_dimension, target_dimension),dtype=data.dtype)
    count =0
    for now_candidate in data:
        current_image = np.reshape(now_candidate, (shape_size[2], shape_size[3]))
        current_image_resized = np.resize(current_image,(target_dimension, target_dimension))
        big_array[count, 0, :, :] = current_image_resized
        big_array[count, 1, :, :] = current_image_resized
        big_array[count, 2, :, :] = current_image_resized

        count = count+1
    return big_array



mnist = mx.test_utils.get_mnist()
batch_size = 100
num_classes =10

target_dimension = 227
big_array_train = resize_batchimage( mnist['train_data'], target_dimension)
big_array_test = resize_batchimage( mnist['test_data'], target_dimension)

#train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
train_iter = mx.io.NDArrayIter(big_array_train[:2000,:,:], mnist['train_label'][:2000], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(big_array_test[:1000,:,:,:], mnist['test_label'][:1000], batch_size)
dtype =mnist['train_data'].dtype
# softmax loss
input_data = mx.sym.Variable(name="data")
if dtype == 'float16':
    input_data = mx.sym.Cast(data=input_data, dtype= np.float16)
# stage 1
conv1 = mx.sym.Convolution(name='conv1',data=input_data, kernel=(11, 11),stride=(4, 4), num_filter=96)
relu1 = mx.sym.Activation(data=conv1, act_type="relu")
lrn1 = mx.sym.LRN(data=relu1, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
pool1 = mx.sym.Pooling(
data=lrn1, pool_type="max", kernel=(3, 3), stride=(2,2))
# stage 2
conv2 = mx.sym.Convolution(name='conv2',
data=pool1, kernel=(5, 5), pad=(2, 2), num_filter=256)
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
lrn2 = mx.sym.LRN(data=relu2, alpha=0.0001, beta=0.75, knorm=2, nsize=5)
pool2 = mx.sym.Pooling(data=lrn2, kernel=(3, 3), stride=(2, 2), pool_type="max")
# stage 3
conv3 = mx.sym.Convolution(name='conv3',
data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=384)
relu3 = mx.sym.Activation(data=conv3, act_type="relu")
conv4 = mx.sym.Convolution(name='conv4',
data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
relu4 = mx.sym.Activation(data=conv4, act_type="relu")
conv5 = mx.sym.Convolution(name='conv5',
data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
relu5 = mx.sym.Activation(data=conv5, act_type="relu")
pool3 = mx.sym.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
# stage 4
flatten = mx.sym.Flatten(data=pool3)
fc1 = mx.sym.FullyConnected(name='fc1', data=flatten, num_hidden=4096)
relu6 = mx.sym.Activation(data=fc1, act_type="relu")
dropout1 = mx.sym.Dropout(data=relu6, p=0.5)
# stage 5
fc2 = mx.sym.FullyConnected(name='fc2', data=dropout1, num_hidden=4096)
relu7 = mx.sym.Activation(data=fc2, act_type="relu")
dropout2 = mx.sym.Dropout(data=relu7, p=0.5)
# stage 6
fc3 = mx.sym.FullyConnected(name='fc3', data=dropout2, num_hidden=num_classes)
if dtype == 'float16':
    fc3 = mx.sym.Cast(data=fc3, dtype=np.float32)
alexnet = mx.sym.SoftmaxOutput(data=fc3, name='softmax')


# create a trainable module on GPU 0
alexnet_model = mx.mod.Module(symbol=alexnet, context=mx.gpu())
# train with the same
alexnet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=10)


test_iter = mx.io.NDArrayIter(big_array_test[:1000,:,:,:], None, batch_size)
prob = alexnet_model.predict(test_iter)
test_iter = mx.io.NDArrayIter(big_array_test[:1000,:,:,:], mnist['test_label'][:1000], batch_size)
# predict accuracy for lenet
acc = mx.metric.Accuracy()
alexnet_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.98


