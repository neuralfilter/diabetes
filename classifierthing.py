import pandas as pd
import numpy as np
import caffe
import os
import glob
import re

net = caffe.Classifier('/home/seiji/caffe/swagnet_deploy.prototxt' , '/home/seiji/Desktop/swaga_iter_50000.caffemodel', mean=np.load('/home/seiji/out.npy'),  channel_swap=(2, 1,0), raw_scale=255, image_dims=(512,512))
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
results = pd.DataFrame(columns=['filename','prediction'])
net.blobs['data'].reshape(1,3,512,512)
n = 0
path = '/home/seiji/Desktop/Data/predictions/test/'
listing1 = os.listdir(path)
listing = sorted(listing1, key=lambda x: (int(re.sub('\D','',x)),x))

for infile in listing:
	input_image = caffe.io.load_image('/home/seiji/Desktop/Data/predictions/test/' + infile)
	prediction = net.predict([input_image], oversample=False)
	print 'predicted class:', prediction
	pred = prediction.argmax()
	results.loc[n] = np.array([infile, pred])
	n+=1
	
results.to_csv("./Desktop/predictions3.csv")