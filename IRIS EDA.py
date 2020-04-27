import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

iris = pd.read_csv(r"C:\Users\rohit\Desktop\Datasets\iris.csv")
#print(iris.shape)
# print(iris.columns)
iris = iris.drop(columns="Id")
#Gives the count of rows per each output class
# print(iris["Species"].value_counts())

# 1D Scatter Plot

iris_setosa = iris.loc[iris["Species"] == "Iris-setosa"]
iris_virginica = iris.loc[iris["Species"] == "Iris-virginica"]
iris_versicolor = iris.loc[iris["Species"] == "Iris-versicolor"]


#Blue points are Setosa, orange points are Versicolor and green points are Virginica
# plt.plot(iris_setosa["PetalLengthCm"],np.zeros_like(iris_setosa["PetalLengthCm"]),'o')
# plt.plot(iris_virginica["PetalLengthCm"], np.zeros_like(iris_virginica["PetalLengthCm"]), 'o')
# plt.plot(iris_versicolor["PetalLengthCm"], np.zeros_like(iris_virginica["PetalLengthCm"]),'o')

# plt.grid()
# plt.show()

# 2D Scatter Plot
# iris.plot(kind = "scatter", x = "SepalLengthCm", y ="SepalWidthCm")
# iris.plot(kind = "scatter", x = "PetalLengthCm", y ="PetalWidthCm")
# plt.show()

#In the above figure, we aren't able to understand which is setosa, which
#is versicolor or virginica because all the points are in the same
#colour.

#Let's try to plot 2-D Scatter Plot with colour for each flower
# sns.set_style("whitegrid")
# sns.FacetGrid(iris, hue="Species", size =4).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()
# plt.show()

#3D Scatter Plot
# import plotly.express as px
# fig = px.scatter_3d(iris, x="SepalLengthCm", y ="SepalWidthCm", z = "PetalWidthCm", color = 'Species')
# fig.show()

"""A 3D plot will be used for three variables or dimensions.
However, what should we do if we have more than 3 dimensions or
features in our dataset as we humans only have the capability
to visualize 3 dimensions. One solution to the problem is pair plots"""

#Pair Plots
#As we have 4 features, we get 4C2 unique plots
# sns.set_style("whitegrid")
# sns.pairplot(iris, hue="Species", size = 3)
# plt.show()

"""Pair plots will only plot the variables which are numberical. The variables which
are of string type, by default, pair plots won't plot automatically. If you want to 
plot, then you need to encode it as numerical. However, Seaborn will encode internally
and assign a label to each unique value in the non-numerical values."""

#Histogram and Introduction to PDF
"""Histogram is an accurate graphical representation of the
distribution of numerical data. PDF is smoothness of histogram"""
# sns.FacetGrid(iris, hue="Species", size =5)\
#     .map(sns.distplot,"PetalLengthCm") \
#     .add_legend()
# plt.show()

#Univariate Analysis using PDF
#Here, we're trying to find which of the 4 features is most important to distinguish between the flowers
#For Petal Width
# sns.FacetGrid(iris,hue = 'Species',size = 5) \
#     .map(sns.distplot, "PetalWidthCm") \
#     .add_legend()
# plt.show()

#For Sepal Width
# sns.FacetGrid(iris, hue="Species",size= 5) \
#     .map(sns.distplot, "SepalWidthCm") \
#     .add_legend()
# plt.show()

#Similarly, we can do Univariate analysis for Petal Length and Sepal Length as well

#Mean, Variance and Standard Deviation
"""
Mean : Average of a given set of data
Variance : Sum of squares of differences between each number and mean divided by the total number of elements
Standard Deviation : Square root of Variance. It is a measure of the extent to which data varies from the mean"""
# Means of petal lengths of different classes
# print(np.mean(iris_setosa["PetalLengthCm"]))
# print(np.mean(iris_virginica["PetalLengthCm"]))
# print(np.mean(iris_versicolor["PetalLengthCm"]))

# Standard Deviation of petal lengths of different classes
# print(np.std(iris_setosa["PetalLengthCm"]))
# print(np.std(iris_virginica["PetalLengthCm"]))
# print(np.std(iris_versicolor["PetalLengthCm"]))

# Median and Quantiles
"""Median: The median is the middle of a sorted list of numbers
Quantile : Any set of data, arranged in ascending or descending order,
can be divided into various partitions or subsets, regulated by quantiles.
Quantile is a generic term for those values that divide the set into 
partitions of size n, so that each part represents 1/n of the set"""

#Box-plot with Whiskers
# sns.boxplot(x="Species", y="PetalLengthCm", data=  iris)
# plt.show()

#Violin Plots
sns.violinplot(x="Species", y="PetalLengthCm", data = iris)
plt.show()