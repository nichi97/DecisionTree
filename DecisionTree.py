#=========================
# Define Constants
#=========================
FEATURE_LEN = 22

#=================
# Helper Methods 
#=================
import numpy as np
from scipy.stats import entropy
from collections import Counter

def computeIG(feature, label, t):
    """ Compute the Informage Gain given a set of feature, label, and critical value. Namely, compute H(X) - H(X|Z)

    Parameters
    ----------
    feature: ndarray, default=None
        The X in H(X|Z). 
    
    label: ndarray, default=None
        The label corresponding to each feature

    t: int, deault=None
        The critical value that specify the decision rule: If X <= t, return 1; if X > t, return 0

    Return
    ------
    The Information Gain from knowing Z
    
    """

    # calculate the marginal prob of x
    x_val_cnt = dict(Counter(label))
    H_x = entropy(list(x_val_cnt.values()))
    
    total_len = len(feature)

    # select from the labels only features that is greater than t, count up how many of them are 0
    p_x_0_z_0 = np.sum(label[feature > t] == 0) / len(label[feature > t])
    p_x_1_z_0 = 1 - p_x_0_z_0

    p_x_0_z_1 = np.sum(label[feature <= t] == 0) / len(label[feature <= t])
    p_x_1_z_1 = 1 - p_x_0_z_1

    p_z_0 = np.sum(feature > t) / total_len
    p_z_1 = np.sum(feature <= t) / total_len

    H_x_z_0 = entropy([p_x_0_z_0, p_x_1_z_0])
    H_x_z_1 = entropy([p_x_0_z_1, p_x_1_z_1])

    return H_x - (p_z_0 * H_x_z_0 + p_z_1 * H_x_z_1)


def findCut(arr):
    """ Find all the places where possible cuts can be made to the decision tree

    Parameters
    ----------
    arr: The feature array
    
    Return
    ------
    array of cuts
    """
    unique_arr = np.unique(arr)
    output = []
    for i in range(len(unique_arr) - 2):
        output.append((unique_arr[i] + unique_arr[i + 1]) / 2)
    return output

def loadData():
    train = np.loadtxt('./data/pa2train.txt')
    train_f = train[:, :FEATURE_LEN]
    train_l = train[:, FEATURE_LEN]

    vali = np.loadtxt('./data/pa2validation.txt')
    vali_f = vali[:, :FEATURE_LEN]
    vali_l = vali[:, FEATURE_LEN]

    test = np.loadtxt('./data/pa2test.txt')
    test_f = test[:, :FEATURE_LEN]
    test_l = test[:, FEATURE_LEN]

    feature_text = np.loadtxt('./data/pa2features.txt', dtype='U')

    return (train, train_f, train_l, vali, vali_f, vali_l, test, test_f, test_l, feature_text)


    
class DecisionTree:

    def __init__(self, data):
        self.root = self.Node()
        self.root.recurseHelper(None, data)
    
    class Node:
        """
        A decision tree node

        Parameters
        ----------
        parent : Node, default=None
            Parent of current node.

        data : nparray, default=None 
            data passes through the current node

        isLeaf : boolean, default=None
            a boolean value that represents whether the current node is a leaf or not

        lc: Node, default=None
            Left child of the current node, representing the YES branch

        rc: Node, default=None
            right child of the current node, representing the NO branch

        feature: int, default=None
            Specifies along which column is the cut happening. 

        t: double, default=None
            Specifies the critical value from which the branches are splitted
        """
        def __init__(self):
            """
            Train the Node when the tree is built
            """
            (self.data, self.parent, self.isLeaf, self.lc, self.rc, self.feature, self.t) = (None, None, False, None, None, None, None)

        def recurseHelper(self, parent, data):
            print("Training Node...")
            (self.data, self.parent) = (data, parent)

            # end of recursion, if the node pure, stop there, assign both lc and rc the output
            if (self.isPure()):
                temp = self.__init__()
                (temp.isLeaf, temp.lc, temp.rc) = (True, data[:, FEATURE_LEN][0], data[:, FEATURE_LEN][0])
                (temp.feature, temp.t) = (None, None)
                print(f"This is a leaf node, outputting {data[:,FEATURE_LEN][0]}")
                return temp 
            else:
                # if the node is not pure, recurse down and initialize lc and rc
                (self.feature, self.t) = self.findOptimalCut()
                self.isLeaf = False

                # if the feature is less than or equal to critical value, that is a YES, corresponds to 1
                # otherwise, it is a no, corresponds to 0
                print(f'Finish training a branch node, with feature = {self.feature}, t = {self.t}')

                self.lc = self.recurseHelper(self, data[data[:, self.feature] <= self.t, :])
                self.rc = self.recurseHelper(data[data[:, self.feature] > self.t, :])


            


        def __str__(self):
            return(f'decision rule: Is X_{self.feature} <= {self.t}?')

   
        def findOptimalCut(self):
            """Find the cut that leads to the maximum information gain from self.data
            
            Returns
            -------

            IG_max_col_idx: the col from which the maximum IG cut is produced
            cut: the cut itself

            """
            feature = self.data[:, :FEATURE_LEN]
            label = self.data[:, FEATURE_LEN]

            # used to compare the best of each column
            IG_max_max = -np.Inf 
            IG_max_col_idx= -1
            IG_max_row_idx = -1
            # loop through each column
            for i in range(feature.shape[1]):
                curr_col = feature[:,i]
                cuts = findCut(curr_col)

                # keep track of index and max IG to get the max IG and its index for a col
                IG_max_idx= -1
                IG_max = -np.inf
                for j, cut in enumerate(cuts):
                    temp = computeIG(curr_col, label, cut)
                    if temp > IG_max:
                        IG_max = temp
                        IG_max_idx = j
                
                # update the column wise max
                if IG_max > IG_max_max:
                    IG_max_max = IG_max
                    IG_max_col_idx = i
                    IG_max_row_idx = IG_max_idx

            # compute the cut again after finding the max IG
            cuts = findCut(feature[:, IG_max_col_idx])
            return (IG_max_col_idx, cuts[IG_max_row_idx])
            
        def isPure(self):
            """ Find out if the value stored under a node is pure or not

            Parameter 
            ---------
            arr: The label array

            Return
            ------
            True if the value is pure, False if the value is not pure
            """

            label = self.data[:, FEATURE_LEN]
            return(len(np.unique(label)) == 1)


if(__name__ == '__main__'):

    # read in the data
    (train, train_f, train_l, vali, vali_f, vali_l, test, test_f, test_l, feature_text) = loadData()
    T = DecisionTree(train)


    
   
