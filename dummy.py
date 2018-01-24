
import mxnet as mx
for i in range(1,100):
    print(i)
    try:
        x= mx.nd.zeros((1,), ctx=mx.gpu(i))
    except:
        print( i)
