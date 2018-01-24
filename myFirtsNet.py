import mxnet as mx
#import logging
import os
import mxnet.log
import time
import numpy as np
import matplotlib.pyplot as plt


def get_my_net():

    data = mx.sym.var('data')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=100)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type= "max", kernel=(2,2), stride=(2,2))
    # first fullc layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
    # softmax loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    #lenet = mx.symbol.SVMOutput(data= fc2, name='softmax',use_linear=False )
    return lenet




    print('Starting prediction on fold {} at trial {} '.format(current_fold, current_trial))
    log_file = open('prediction_trial_' + str(current_trial) + '.txt', 'a')
    log_file.write('Prediction on fold {}\n'.format(current_fold))

def get_meaner_and_max_epoch(current_fold, current_trial):
    arr = []
    meaner = 0
    with open('process_fold_' + str(current_fold) + '_trial_' + str(current_trial) +  '.log', 'r') as f:
        for line in f:
            if line.find('Validation-accuracy=') > 0:
                epoch = line[line.find('[') + 1:line.find(']')]
                my_bac = line[line.find('=') + 1:]
                #print('Epoch:', epoch, 'my_bac: ', my_bac)
                arr.append([epoch, float(my_bac)])

    arr = np.array(arr)
    max_epoch = arr.argmax(axis=0)[1] # get index of maximal validation result
    max_epoch_validation = arr[max_epoch][1] # get validation result of max epoch
    max_epoch += 1 # add 1 to fix indexing issue
    return arr

if __name__ == "__main__":
    mnist = mx.test_utils.get_mnist()
    batch_size = 100
    #os.chdir('C:\D')  # Change to tmp dir, more space on drive D

    print(os.getcwd())

    # log_file = '1.log'#''process_fold_' + str(0) + '_trial_' + str(1) + '.log'
    # logging.basicConfig(format='%(asctime)s %(levelname)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p', filename=log_file, level=logging.INFO)
    # logging.info('Started training on fold {} at trial {}'.format(0, 0))
    current_fold  =0
    current_trial =0
    logfilenamer = 'process_fold_' + str(current_fold) + '_trial_' + str(current_trial)+'.log'
    logzulu = mxnet.log.get_logger(name='kaka', filename=logfilenamer, level=mxnet.log.DEBUG,
                                   filemode='w')
    logzulu.error('what the hell ' + time.strftime('%x %X'))

    #logging.getLogger().setLevel(logging.INFO)  # to show the output log during training

    train_iter = mx.io.NDArrayIter(mnist['train_data'],mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'],
                                 batch_size)  # important as the prediction need not have equal barch size
    lenet =get_my_net()
    # create a trainable module on GPU 0
    lenet_model = mx.mod.Module(symbol=lenet, context=mx.gpu(),logger=logzulu)
    # train with the same
    lenet_model.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params={'learning_rate':0.1},
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                    num_epoch=10,
                    initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))


    test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
    prob = lenet_model.predict(test_iter)
    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    array_validation = get_meaner_and_max_epoch(current_fold, current_trial)
    plt.figure(1)
    plt.plot(array_validation[:,0], array_validation[:,1], 'ro')

    # print('Starting prediction on fold {} at trial {} '.format(current_fold, current_trial))
    # log_file = open('prediction_trial_' + str(current_trial) + '.txt', 'a')
    # log_file.write('Prediction on fold {}\n'.format(current_fold))


    acc = mx.metric.Accuracy()
    lenet_model.score(test_iter, acc)
    print('the accuracy is')
    print(acc)
#assert acc.get()[1] > 0.98

