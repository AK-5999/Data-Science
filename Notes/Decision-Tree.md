# Part 1 – Decision Tree Classifier: In-Depth Intuition

**YouTube Link:** https://www.youtube.com/watch?v=ynTCUngbFHA

## 🧠 Topic Overview : Decision Tree Classifiers

This focuses on how decision trees divide data based on features to perform classification tasks.


## 📌 Key Concepts Covered

### 🔹 What Is a Decision Tree Classifier?
A model that recursively splits data into subsets based on feature values to group similar target outcomes.

### 🔹 Nodes Explained
- **Root Node**: The starting point of the tree.
- **Decision Nodes**: Points where data is split based on feature conditions.
- **Leaf Nodes**: Final nodes that represent predicted class labels.

### 🔹 Splitting Criteria

- **Entropy**
- **Gini Impurity**


These measure how “pure” a dataset subset is, which guides the selection of the best split.

### 🔹 Feature Selection

- **Information Gain**


These measure how features should be selected.

<img width="270" height="186" alt="image" src="https://github.com/user-attachments/assets/2ef13a96-0022-43ec-913b-8926f5d8eb56" />


## 🧩 Intuition Behind the Learning Process

- The classifier starts with all training data.
- At each node, it identifies the split that best separates classes.
- It continues splitting until nodes are pure or a stopping condition is met.

## Note:
Decision tree for both classification regression here classification. 2 techniqiues. ID3 and CART. 
In Cart the decision tress spilts into binary trees. 
- a) Entropy and Gini Index(Purity spirit) 
- b) Information Gain(feature decision tree split). 
To check for Pure split two techniques called Entropy and Gini impurity are used and INformation gain (how the features are selected) is used.

The range of entropy remains between 0 to 1.
When H(S) is zero then that is pure split. And when H(s) is 1 then that is impure split ie equal distribution (eg 3yes and 3nos).
The gini impurity ranges between 0 to 0.5. 
In impure split the Gini impurity comes out to be 0.5 and in pure split it is 0. 

gini impurity is preferrable over entropy because of involvement of log it may slow down.

Now if you have multiple features, you use information gain to know how to make the tree using the given features whether which feature will start and which one will follow later. The feature starting with which the information gain calculation comes out to be the most should be the one with which the decision tree should be started.
