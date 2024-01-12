# ECE 1395
# Problem Set 1
# Beryl Sin

# imports
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

#----------------------------------QUESTION-3-PART-A---------------------------------------------
# seed random number generator
np.random.seed(1)
# generate a 1000000x1 vector x of random numbers from a Gaussian distribution
vector_x = np.random.randn(1000000)
# scale vector x to ensure a mean of 1.5 with a standard deviation of 0.6
# using eqn: y = m_2 + (x - m_1) * s_2 / s_1
vector_x = 1.5 + (vector_x - np.mean(vector_x)) * (0.6 / np.std(vector_x)) 

#----------------------------------QUESTION-3-PART-B---------------------------------------------
# generate a 1000000x1 vector z of random numbers from a uniform distribution 
# with distribution [-1, 3]
vector_z = np.random.uniform(-1, 3, 1000000)

#----------------------------------QUESTION-3-PART-C---------------------------------------------
# normalized histogram for vector x
hist_x = plt.figure(1)
plt.hist(vector_x, bins=50, color='Pink', edgecolor='Black', density=True)
plt.title('Vector X')
plt.savefig('output/ps1-3-c-1.png') # output

hist_x.show()

# normalized histogram for vector y
hist_z = plt.figure(2)
plt.hist(vector_z, bins=50, color='Cyan', edgecolor='Black', density=True)
plt.title('Vector Z')
plt.savefig('output/ps1-3-c-2.png') # output

#----------------------------------QUESTION-3-PART-D---------------------------------------------
# record start time
start_time = datetime.now()
# add 1 to each element in vector x using for loop
for i in itertools.product(*[range(s) for s in vector_x.shape]):
    vector_x[i] += 1
# calculate and print execution time
end_time = datetime.now() - start_time
print('Question 3D execution time in hh:mm:ss : ', end_time)


#----------------------------------QUESTION-3-PART-E---------------------------------------------
# record start time
start_time = datetime.now()
# add 1 to each element in vector x without using a loop
vector_x += 1
# calculate and print execution time
end_time = datetime.now() - start_time
print('Question 3E execution time in hh:mm:ss : ', end_time)

#----------------------------------QUESTION-3-PART-F---------------------------------------------
# generate vector y containing elements that are numbers in vector z that are > 0 and < 1.5
vector_y = [] # will be 1D
for i in vector_z:
    if i > 0 and i < 1.5:
        vector_y.append(i)
# print out number of elements in vector y
print('Question 3F # elements :', len(vector_y))

#----------------------------------QUESTION-4-PART-A---------------------------------------------
# define matrix a
matrix_a = np.array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])
# min value of each column using infinity-norm
temp_array = 1. / matrix_a # inverse
c0_min = 1. / np.linalg.norm(temp_array[:, 0], np.inf)
c1_min = 1. / np.linalg.norm(temp_array[:, 1], np.inf)
c2_min = 1. / np.linalg.norm(temp_array[:, 2], np.inf)
# max value of each row using infinity-norm
r0_max = np.linalg.norm(matrix_a[0, :], np.inf)
r1_max = np.linalg.norm(matrix_a[1, :], np.inf)
r2_max = np.linalg.norm(matrix_a[2, :], np.inf)
# highest value
temp_array = np.array([[r0_max, r1_max, r2_max]])
max_value = np.linalg.norm(temp_array[0, :], np.inf)
# sum of each column using 1-norm
c0_sum = np.linalg.norm(matrix_a[:, 0], 1)
c1_sum = np.linalg.norm(matrix_a[:, 1], 1)
c2_sum = np.linalg.norm(matrix_a[:, 2], 1)
# sum of all elements
temp_array = np.array([[c0_sum, c1_sum, c2_sum]])
total_sum = np.linalg.norm(temp_array[0, :], 1)
# define matrix b with the squared value of each element in matrix a
matrix_b = matrix_a**2
# print out results
print('Question 4A min values : ', c0_min, c1_min, c2_min)
print('Question 4A max values : ', r0_max, r1_max, r2_max)
print('Question 4A highest value : ', max_value)
print('Question 4A sums : ', c0_sum, c1_sum, c2_sum)
print('Question 4A total sum : ', total_sum)
print('Question 4A matrix b : ', matrix_b)

#----------------------------------QUESTION-4-PART-B---------------------------------------------
# define coefficient matrix
matrix_a = np.array([[2, 1, 3], [2, 6, 8], [6, 8, 18]])
# define solution matrix
matrix_b = np.array([[1], [3], [5]])
# solve for unknown matrix through inverse method, x = A^-1 * B
matrix_x = np.dot(np.linalg.inv(matrix_a), matrix_b)
# x = matrix_x[0], y = matrix_x[1], z = matrix_x[2]
print('Question 4B x value : ', matrix_x[0])
print('Question 4B y value : ', matrix_x[1])
print('Question 4B z value : ', matrix_x[2])

#----------------------------------QUESTION-4-PART-C---------------------------------------------
# define vector 1
vector_one = np.array([[-0.5, 0, 1.5]])
# define vector 2
vector_two = np.array([[-1, 1, 0]])
# find 1-norm of each vector, adding all the magnitude values of each element
x1_1norm = np.linalg.norm(vector_one[0, :], 1)
x2_1norm = np.linalg.norm(vector_two[0, :], 1)
# find 2-norm of each vector, distance - sqrt of sum of squared values of each element
x1_2norm = np.linalg.norm(vector_one[0, :], 2)
x2_2norm = np.linalg.norm(vector_two[0, :], 2)
# print results
print('Question 4C x1 1-norm value : ', x1_1norm)
print('Question 4C x2 1-norm value : ', x2_1norm)
print('Question 4C x1 2-norm value : ', x1_2norm)
print('Question 4C x2 2-norm value : ', x2_2norm)

