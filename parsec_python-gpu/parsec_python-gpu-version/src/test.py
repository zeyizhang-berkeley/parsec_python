import numpy as np
from numpy.linalg import eigh
# from splineData import splineData
# import matplotlib.pyplot as plt
#
# AtomFuncData, data_list = splineData()
# atomic_num = 2
#
# radius = AtomFuncData[atomic_num]['data'][:,0]
# charge = AtomFuncData[atomic_num]['data'][:,1]
# hartree = AtomFuncData[atomic_num]['data'][:,2]
# pot_P = AtomFuncData[atomic_num]['data'][:,3]
# pot_S = AtomFuncData[atomic_num]['data'][:,4]
# wfn_P = AtomFuncData[atomic_num]['data'][:,5]
# wfn_S = AtomFuncData[atomic_num]['data'][:,6]
#
# plt.figure(figsize=(10, 6))
# size = 0.01
# # charge vs radius
# plt.plot(radius, charge, label='charge', marker='x', markersize = size)
# # Add labels and title
# plt.xlabel('radius', fontsize=14)
# plt.ylabel('charge', fontsize=14)
# plt.title('charge vs radius', fontsize=16)
# # Add a legend
# plt.legend(fontsize=12)
# # Display the plot
# plt.grid(True)
# plt.show()
#
# # hartree vs radius
# plt.plot(radius, hartree, label='hartree', marker='x', markersize = size)
# # Add labels and title
# plt.xlabel('radius', fontsize=14)
# plt.ylabel('hartree', fontsize=14)
# plt.title('hartree vs radius', fontsize=16)
# # Add a legend
# plt.legend(fontsize=12)
# # Display the plot
# plt.grid(True)
# plt.show()
#
# # pot_P vs radius
# plt.plot(radius, pot_P, label='pot_P', marker='x', markersize = size)
# # Add labels and title
# plt.xlabel('radius', fontsize=14)
# plt.ylabel('pot_P', fontsize=14)
# plt.title('pot_P vs radius', fontsize=16)
# # Add a legend
# plt.legend(fontsize=12)
# # Display the plot
# plt.grid(True)
# plt.show()
#
# # pot_S vs radius
# plt.plot(radius, pot_S, label='pot_S', marker='x', markersize = size)
# # Add labels and title
# plt.xlabel('radius', fontsize=14)
# plt.ylabel('pot_S', fontsize=14)
# plt.title('pot_S vs radius', fontsize=16)
# # Add a legend
# plt.legend(fontsize=12)
# # Display the plot
# plt.grid(True)
# plt.show()
#
# # wfn_P vs radius
# plt.plot(radius, wfn_P, label='wfn_P', marker='x', markersize = size)
# # Add labels and title
# plt.xlabel('radius', fontsize=14)
# plt.ylabel('wfn_P', fontsize=14)
# plt.title('wfn_P vs radius', fontsize=16)
# # Add a legend
# plt.legend(fontsize=12)
# # Display the plot
# plt.grid(True)
# plt.show()
#
# # wfn_S vs radius
# plt.plot(radius, wfn_S, label='wfn_S', marker='x', markersize = size)
# # Add labels and title
# plt.xlabel('radius', fontsize=14)
# plt.ylabel('wfn_S', fontsize=14)
# plt.title('wfn_S vs radius', fontsize=16)
# # Add a legend
# plt.legend(fontsize=12)
# # Display the plot
# plt.grid(True)
# plt.show()
#
# #print(AtomFuncData[0]['data'][:,0])

DX = np.array([])
m = DX.shape[1]
print(m)