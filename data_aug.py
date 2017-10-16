import numpy as np
import os

f = np.rot90
rot3 = (1,2)
rot2 = (2,0)
rot1 = (0,1)
g = np.flip

functions = []
functions.append(lambda A: A);
functions.append(lambda A: g(A,axis = 2));
functions.append(lambda A: f(f(A,2,rot3),3,rot2));
functions.append(lambda A: g(f(f(A,2,rot3),3,rot2),axis = 2));
functions.append(lambda A: f(A,1,rot3));
functions.append(lambda A: g(f(A,1,rot3),axis = 2));
functions.append(lambda A: f(f(A,3,rot3),3,rot1));
functions.append(lambda A: g(f(f(A,3,rot3),3,rot1),axis = 2));
functions.append(lambda A: f(A,3,rot3));
functions.append(lambda A: g(f(A,3,rot3),axis = 2));
functions.append(lambda A: f(f(A,1,rot3),1,rot1));
functions.append(lambda A: g(f(f(A,1,rot3),1,rot1),axis = 2));
functions.append(lambda A: f(A,2,rot3));
functions.append(lambda A: g(f(A,2,rot3),axis = 2));
functions.append(lambda A: f(A,1,rot2));
functions.append(lambda A: g(f(A,1,rot2),axis = 2));
functions.append(lambda A: f(A,3,rot1));
functions.append(lambda A: g(f(A,3,rot1),axis = 2));
functions.append(lambda A: f(f(A,1,rot3),3,rot2));
functions.append(lambda A: g(f(f(A,1,rot3),3,rot2),axis = 2));
functions.append(lambda A: f(f(A,1,rot3),1,rot2));
functions.append(lambda A: g(f(f(A,1,rot3),1,rot2),axis = 2));
functions.append(lambda A: f(f(A,3,rot1),2,rot2));
functions.append(lambda A: g(f(f(A,3,rot1),2,rot2),axis = 2));
functions.append(lambda A: f(f(A,2,rot3),1,rot1));
functions.append(lambda A: g(f(f(A,2,rot3),1,rot1),axis = 2));
functions.append(lambda A: f(f(A,3,rot3),1,rot2));
functions.append(lambda A: g(f(f(A,3,rot3),1,rot2),axis = 2));
functions.append(lambda A: f(A,1,rot1));
functions.append(lambda A: g(f(A,1,rot1),axis = 2));
functions.append(lambda A: f(f(A,3,rot3),3,rot2));
functions.append(lambda A: g(f(f(A,3,rot3),3,rot2),axis = 2));
functions.append(lambda A: f(f(A,2,rot2),3,rot3));
functions.append(lambda A: g(f(f(A,2,rot2),3,rot3),axis = 2));
functions.append(lambda A: f(f(A,3,rot1),3,rot2));
functions.append(lambda A: g(f(f(A,3,rot1),3,rot2),axis = 2));
functions.append(lambda A: f(A,2,rot1));
functions.append(lambda A: g(f(A,2,rot1),axis = 2));
functions.append(lambda A: f(A,3,rot2));
functions.append(lambda A: g(f(A,3,rot2),axis = 2));
functions.append(lambda A: f(f(A,1,rot1),3,rot2));
functions.append(lambda A: g(f(f(A,1,rot1),3,rot2),axis = 2));
functions.append(lambda A: f(f(A,1,rot3),2,rot1));
functions.append(lambda A: g(f(f(A,1,rot3),2,rot1),axis = 2));
functions.append(lambda A: f(A,2,rot2));
functions.append(lambda A: g(f(A,2,rot2),axis = 2));
functions.append(lambda A: f(f(A,1,rot2),2,rot1));
functions.append(lambda A: g(f(f(A,1,rot2),2,rot1),axis = 2));

'''
for i in range(5,100):
        print 'i = '+str(i)
	os.chdir('/zfsauton/home/siyuh/256_64/'+str('01')+str(i).rjust(3,'0'))
        print '/zfsauton/home/siyuh/256_64/'+str('01')+str(i).rjust(3,'0')
	for j in range(0,64):
		print j
		data = np.squeeze(np.load(str(j)+'.npy'))
                num = 0
                for h in functions:
			data_aug = np.expand_dims(h(data), axis=4)
                        fileNum = 64+j*48+num
                        np.save(str(fileNum),data_aug)
			num = num+1
'''             
