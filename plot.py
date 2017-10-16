import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = np.loadtxt('losses.txt')
plt.figure()
plt.plot(x)
plt.ylabel('loss')
plt.savefig('loss.png')


x = np.loadtxt('loss_val_batch20.txt')
#y = np.loadtxt('loss_train.txt')
plt.figure()
plt.plot(x,label='validation')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend()
plt.savefig('test.png')

plt.figure()
ax1 = plt.subplot(121,aspect='equal')
ax2 = plt.subplot(122,aspect='equal')
ax1.set_ylabel(r'$\Omega_m$:pred')
ax1.set_xlabel(r'$\Omega_m$:true')
ax1.set_xlim([0.22,0.42])
ax1.set_ylim([0.22,0.42])
ax2.set_ylabel(r'$\sigma_8$:pred')
ax2.set_xlabel(r'$\sigma_8$:true')
ax2.set_xlim([0.75,1.05])
ax2.set_ylim([0.75,1.05])

for i in range(65,70):
	x = np.loadtxt('pred/val_pred'+str(i)+'.txt')
#y = np.loadtxt('label.txt')
	ax1.scatter(x[:,0],x[:,2],s=2,edgecolor='red')
	ax2.scatter(x[:,1],x[:,3],s=2,edgecolor='red')
#plt.tight_layout()
#ax1.axis('equal')
#ax2.axis('equal')
plt.savefig('compare.png')

