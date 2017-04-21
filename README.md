# Discriminitive Analysis
An implementation of QDA and LDA in Python3 using only numpy.

I made this program for my Machine Learning course taught by Yi Fang at Santa Clara University in the spring of 2017. It trains a Gaussian model based on the input to categorize data via a quadratic or linear discrimitive analysis function. Specifically, the assignment asked to categorize flowers based on 4 features, but the code is written in a way to support any data set in a CSV file. 

It also reports which features were not useful in categorizing the data, which were linearly seperable, and does the analysis with regular covariance matricies as well as diagonal matricies. 

The program can also shuffle the data points around so that each run of the program produces semi-unique output. I found this to be useful because without shuffling, the program predicts the test data with 0% error. Strange result, but very possible with a data set of only 150 points. Shuffling the input makes the error rate greater than 0%.

The assignment, as well as my responses have been included in this repo. See Assignment.pdf, and AssignmentResponse.pdf.
