# KNN-algorithm-based-on-watermelon-dataset
Implementation of KNN algorithm based on watermelon data set
## Author: Rui Zhu
### Creation_time: 2020.11.11
### Content: KNN algorithm implementation based on watermelon dataset
### Blog: https://zhu-rui.blog.csdn.net/


### K-nearest neighbor method
- The k-nearest neighbour (k-NN) method is a basic classification and regression method. This book only discusses the k-nearest neighbor method for classification problems. k-nearest neighbor method has as input the feature vector of instances, corresponding to the points in the feature space, and as output the class of instances, which can be taken as multiple classes. k-nearest neighbor method assumes a training data set, in which the classes of instances are already determined. For classification, new instances are predicted based on the categories of their k nearest neighbour training instances, e.g. by majority voting. Thus, the k-nearest neighbour method does not have an explicit learning process. k-nearest neighbours actually uses the training dataset to partition the feature vector space and act as a 'model' for its classification.

- The choice of k-value, the distance metric and the classification decision rule are the three basic elements of the k-nearest neighbour method. k-nearest neighbour was proposed by Cover and Hart in 1968. This chapter first describes the k-nearest neighbour algorithm, then discusses the model and the three basic elements of the k-nearest neighbour method, and finally describes one implementation of the k-nearest neighbour method, the kd tree, and introduces the algorithm for constructing the kd tree and searching the kd tree.

- K-nearest neighbour algorithm idea: given a training dataset, for a new input instance, find the k instances in the training dataset that are closest to the instance, and the majority of these k instances belong to a certain class, and classify the input instance into that class.

- K-nearest neighbour model: The model used in the k-nearest neighbour method actually corresponds to a partitioning of the feature space. The model is determined by three basic elements - the distance metric, the choice of k-value and the classification decision rule.
- Distance metric: The distance between two instance points in the feature space is a reflection of the degree of similarity between the two instance points. k-nearest neighbour models have a feature space that is typically an n-dimensional real vector space Rn. The distance used is the Euclidean distance, but other distances can be used, such as the more general Lp distance (Lp distance) or the Minkowski distance (Minkowski distance).
- Classification decision rule: The classification decision rule in the k-nearest neighbour method is often majority voting, i.e. the majority class of the k neighbouring training instances of the input instance determines the class of the input instance.

- Implementation of K-nearest neighbor method: the algorithm idea is simple, but the computational complexity is high, so an efficient algorithm KD tree has been proposed

In the k-nearest neighbour method, once the training set, the distance metric (e.g. Euclidean distance), the k-value and the classification decision rule (e.g. majority voting) have been determined, the class to which it belongs is uniquely determined for any new input instance. This corresponds to dividing the feature space into a number of subspaces based on the above elements and determining the class to which each point in the subspace belongs. This fact can be seen clearly in the nearest neighbour algorithm. In the feature space, for each training instance point ix, all points closer to that point than to other points form a region called a cell. Each training instance point has a cell, and the cells of all training instance points form a partition of the feature space. The nearest neighbour method uses the class iy of an instance ix as the class label for all points in its cell. In this way, the class of instance points is determined for each cell.
