# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.io as sio
from scipy.spatial.distance import cdist
from tabulate import tabulate
from sklearn.metrics import accuracy_score

t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
left = [112, 112, 112, 112, 112, 112, 112, 112, 112, 112]
right = [104, 103, 103, 103, 103, 103, 103, 103, 103, 104]
plot1 = plt.figure(1)
plt.scatter(t, left, c='tab:blue', marker='o', label='Left Motor')
plt.scatter(t, right, c='tab:cyan', marker='o', label='Right Motor')
plt.xlabel('Time (s)')
plt.ylabel('Average Rotations Per Minute')
plt.title('Average Tachometer Readings')
plt.legend()
plt.show()

t = [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 11.9, 12.9, 13.9, 14.9, 15.9, 16.9, 17.9, 18.9, 19.9, 20.9, 21.9, 22.9, 23.9, 24.9, 25.9, 26.9, 27.9, 28.9, 29.9, 30.9, 31.9, 32.9, 33.9, 34.9, 35.9, 36.9, 37.9]
rpm_l = [0, 48, 51, 48, 49, 50, 49, 49, 50, 51, 51, 52, 51, 53, 52, 53, 52, 51, 52, 51, 52, 51, 51, 51, 52, 52, 53, 53, 54, 54, 53, 54, 53, 54, 54, 54, 54, 53]
error_l = [65, 17, 14, 17, 16, 15, 16, 16, 15, 14, 14, 13, 14, 12, 13, 12, 13, 14, 13, 14, 13, 14, 14, 14, 13, 13, 12, 12, 11, 11, 12, 11, 12, 11, 11, 11, 11, 12]

rpm_r = [0, 44, 45, 45, 46, 45, 46, 45, 46, 45, 46, 46, 46, 47, 47, 49, 47, 48, 47, 46, 47, 46, 46, 45, 47, 46, 48, 47, 48, 47, 46, 48, 47, 46, 47, 46, 47, 46]
error_r = [85, 41, 40, 40, 39, 40, 39, 40, 39, 40, 39, 39, 39, 38, 38, 36, 38, 37, 38, 39, 38, 39, 39, 40, 38, 39, 37, 38, 37, 38, 39, 37, 38, 39, 38, 39, 38, 39]
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(t, rpm_l, 'g-')
ax2.plot(t, error_l, 'b-')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Rotations Per Minute', color='g')
ax2.set_ylabel('Error', color='b')
plt.title('Responsive Values For Integral Gain (Left Motor)')
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t, rpm_r, 'g-')
ax2.plot(t, error_r, 'b-')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Rotations Per Minute', color='g')
ax2.set_ylabel('Error', color='b')
plt.title('Responsive Values For Integral Gain (Right Motor)')
plt.show()

t = [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 11.9, 12.9, 13.9, 14.9, 15.9, 16.9, 17.9, 18.9, 19.9, 20.9, 21.9, 22.9, 23.9, 24.9, 25.9, 26.9, 27.9, 28.9, 29.9, 30.9, 31.9, 32.9, 33.9, 34.9, 35.9, 36.9, 37.9]
rpm_l = [0, 43, 43, 43, 42, 43, 44, 45, 44, 44, 44, 45, 43, 45, 47, 47, 46, 48, 48, 47, 48, 50, 50, 49, 50, 52, 51, 52, 54, 53, 54, 56, 56, 56, 56, 58, 58, 59]
error_l = [65, 22, 22, 22, 23, 22, 21, 20, 21, 21, 21, 20, 22, 20, 18, 18, 19, 17, 17, 18, 17, 15, 15, 16, 15, 13, 14, 13, 11, 12, 11, 9, 9, 9, 9, 7, 7, 6]

rpm_r = [0, 38, 39, 39, 38, 38, 39, 39, 38, 38, 38, 39, 39, 39, 40, 41, 39, 40, 38, 39, 40, 42, 42, 41, 40, 42, 40, 42, 43, 43, 42, 43, 43, 43, 43, 45, 43, 44]
error_r = [85, 47, 46, 46, 47, 47, 46, 46, 47, 47, 47, 46, 46, 46, 45, 44, 46, 45, 47, 46, 45, 43, 43, 44, 45, 43, 45, 43, 42, 42, 43, 42, 42, 42, 42, 40, 42, 41]
print(len(rpm_r))
print(len(error_r))

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(t, rpm_l, 'g-')
ax2.plot(t, error_l, 'b-')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Rotations Per Minute', color='g')
ax2.set_ylabel('Error', color='b')
plt.title('Responsive Values For PID Gains (Left Motor)')
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(t, rpm_r, 'g-')
ax2.plot(t, error_r, 'b-')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Rotations Per Minute', color='g')
ax2.set_ylabel('Error', color='b')
plt.title('Responsive Values For PID Gains (Right Motor)')
plt.show()


distance = [50, 100, 150, 200, 250, 300, 350, 400]
reading0 = [95, 139, 166, 240, 301, 349, 404, 427]
reading1 = [95, 142, 166, 241, 297, 348, 402, 434]
reading2 = [96, 137, 166, 239, 292, 347, 401, 433]
reading3 = [95, 142, 166, 240, 296, 348, 402, 430]
reading4 = [95, 137, 166, 240, 300, 348, 404, 431]

plot3 = plt.figure(3)
plt.scatter(distance, reading0, c='b', marker='o', label='1')
plt.scatter(distance, reading1, c='g', marker='o', label='2')
plt.scatter(distance, reading2, c='r', marker='o', label='3')
plt.scatter(distance, reading3, c='c', marker='o', label='4')
plt.scatter(distance, reading4, c='m', marker='o', label='5')

plt.xlabel('Distance (mm)')
plt.ylabel('Center Channel Measurement')
plt.title('Measured Distance vs Sensor Output')
plt.legend()
plt.show()