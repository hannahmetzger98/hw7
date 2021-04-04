import pandas as pd
import scipy.spatial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
from sklearn import model_selection
import seaborn as sns
# -----------------------------------------------------------------------------
# From hw06

def findNearestHOF(df, testRow):
    s = df.apply(lambda row: scipy.spatial.distance.euclidean(row.loc[:], testRow), axis = 1)
    minID = s.idxmin()
    return minID

def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    
    # actualOutputSeries == predOutputSeries makes a Series of Boolean values.
    # So in this case, value_counts() makes a Series with just two elements:
    # - with index "False" is the number of times False appears in the Series
    # - with index "True" is the number of times True appears in the Series

    # print("compare:", compare, type(compare), sep='\n', end='\n\n')
    
    # if there are no Trues in compare, then compare[True] throws an error. So we have to check:
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0
    
    return accuracy

class OneNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.inputsDF = None
        self.outputSeries = None
        self.scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True) 
    
    def fit(self, inputsDF, outputSeries):
        self.inputsDF = inputsDF
        self.outputSeries = outputSeries
        return self
    
    def predict(self, testInput):
         if isinstance(testInput, pd.core.series.Series):
            # testInput is a Series, so predict for just this one row
             s = self.inputsDF.apply(lambda row: findNearestHOF(self.inputsDF,testInput), axis =1)
             minID = findNearestHOF(self.inputsDF,testInput)
             s2 = s.map(lambda nearestInputIndx:  self.outputSeries.loc[nearestInputIndx])
             return s2.loc[minID]
            #return s.map(lambda nearestTrainIndx: trainOutputSeries.loc[nearestTrainIndx])
         else:
            s = testInput.apply(lambda row: findNearestHOF(self.inputsDF,row), axis =1)
            s2 = s.map(lambda nearestInputIndx:  self.outputSeries.loc[nearestInputIndx])
            return s2


# -----------------------------------------------------------------------------
# Problem 1

# Given
def operationsOnDataFrames():
    d = {'x' : pd.Series([1, 2], index=['a', 'b']),
         'y' : pd.Series([10, 11], index=['a', 'b']),
         'z' : pd.Series([30, 25], index=['a', 'b'])}
    df = pd.DataFrame(d)
    print("Original df:", df, type(df), sep='\n', end='\n\n')
    
    cols = ['x', 'z']
    
    df.loc[:, cols] = df.loc[:, cols] / 2
    print("Certain columns / 2:", df, type(df), sep='\n', end='\n\n')
    
    maxResults = df.loc[:, cols].max()
    print("Max results:", maxResults, type(maxResults), sep='\n', end='\n\n')
    
# Given
def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  # concatenate two lists into one
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    
    # Need to mix this up before doing CV
    # wineDF = wineDF.sample(frac=1)  # "sample" 100% randomly.
    wineDF = wineDF.sample(frac=1, random_state=99).reset_index(drop=True)
    
    return wineDF, inputCols, outputCol

# Given
def testStandardize():
    df, inputCols, outputCol = readData()
    someCols = inputCols[2:5]
    print("Before standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    standardize(df, someCols)
    print("After standardization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    
    # Proof of standardization:
    print("Means are approx 0:", df.loc[:, someCols].mean(), sep='\n', end='\n\n')
    print("Stds are approx 1:", df.loc[:, someCols].std(), sep='\n', end='\n\n')

# is our magnesisum value for the mean concerning? 
def standardize(df, lc):
    df.loc[:, lc] = StandardScaler().fit_transform(df.loc[:, lc])
    return df


	
	
# -----------------------------------------------------------------------------
# Problem 2

# Given
def testNormalize():
    df, inputCols, outputCol = readData()
    someCols = inputCols[2:5]
    print("Before normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    normalize(df, someCols)
    print("After normalization, first 5 rows:", df.loc[:, inputCols[1:6]].head(5), sep='\n', end='\n\n')
    
    # Proof of normalization:
    print("Maxes are 1:", df.loc[:, someCols].max(), sep='\n', end='\n\n')
    print("Mins are 0:", df.loc[:, someCols].min(), sep='\n', end='\n\n')

def normalize(df, lc):
    df.loc[:, lc] = ((df.loc[:, lc] - df.loc[:, lc].min())/(df.loc[:, lc].max() - df.loc[:, lc].min()))
    return df
 
    
 



    
# -----------------------------------------------------------------------------
# Problem 3

def comparePreprocessing():
    
    df, inputCols, outputCol = readData()
    inputDF = df.loc[:, inputCols]
    outputSeries = df.loc[:, outputCol]
    
    #Orignal dataset, k folds, 10-fold cross validation on the dataset, loop executes 1o times,
    #make training and testing, fit testing and predict on training, do this 10 times. 
    #hw5 did this manual, hw6 still built training and testing, did the 1NN test find the nearest row and returned
    #classification 
    #class, fit, lets model learn and predict does the predications on the stuff we wanted it to. 
    #see 
    
    og = OneNNClassifier()
    cvScores = model_selection.cross_val_score(og, inputDF, outputSeries, cv=10, scoring=og.scorer)
    print(cvScores.mean())
    
    
    #norm dataset, before you normalize make a copy; dont use testNorm
    #use normalize, and then use that returned df to call crossvalScore
    dfN = df.copy()
    dfN, inputCols, outputCol = readData()
    someCols = inputCols[:]
    dfN = normalize(dfN, someCols)
    
    inputDF = dfN.loc[:, inputCols]
    outputSeries = dfN.loc[:, outputCol]
    
    n = OneNNClassifier()
    cvScoresN = model_selection.cross_val_score(n, inputDF, outputSeries, cv=10, scoring = n.scorer)
    print(cvScoresN.mean())
    

    
    #stan dataset
    dfS = df.copy()
    dfS, inputCols, outputCol = readData()
    someCols= inputCols[:]
    dfS = standardize(dfS, someCols)
    
    
    inputDFS = dfS.loc[:, inputCols]
    outputsSerS = dfS.loc[:, outputCol]
    
    s = OneNNClassifier()
    cvScoresS = model_selection.cross_val_score(s, inputDFS, outputsSerS, cv=10, scoring = s.scorer)
    print(cvScoresS.mean())
    
    '''
    a. the results: the first is the fastest becuase it does not require calling a stand/norm function.
    the other two, standardize and normalize are really close, because they require similar computing energy
    from the CPU to excute the math to change the data in their structures. 
    
    b. z-transformation essentially scales dissimilar data, so like in the wine, some of the colms have numbers
    in the 100's and others in 2.33 digits. so then you can compare the scores of that data. 1NN 96.1% (z-transformed data))
    
    c. leave-one-out is a special case of cross validatoin, where only one instance of the data is used as the 
    testing set at a time, so it goes through the whole set of data using a bunch of training information,
    and that would report a higher accuracy for the results. 
            0.7584967320261438
            0.9490196078431372
            0.9545751633986927
    
    '''
# -----------------------------------------------------------------------------
# Problem 4

'''
a. -0.05148233107713217
it is negatively skewed
b. this is a total guess, but (0, -.5)
c. most likely classification 1. 
d. because Diluted and Proline have some difference between the classifications in their hisotgrams, while only 
keepign the two would skew data alot, we think these two may help still represent a deceent accuracy. 
so itd be skewed but maybe these wouldnt add horribly to skewing. 
e.Nonflavanoid Phenols and Ash, ... lets ask tm. the ash i will say has most of its classification in the 0 zone
so none of the three classes skewed more positively or negativley. 

Diluted and proline: 0.8764705882352942, 
ash and n acids: 0.4879084967320261


'''
def testSubsets():
    dfS, inputCols, outputCol = readData()
    
    inputCols = ["Diluted", "Proline"]
    someCols = inputCols
    dfS = standardize(dfS, someCols)
    
    
    inputDFS = dfS.loc[:, someCols]
    outputsSerS = dfS.loc[:, outputCol]
    
    s = OneNNClassifier()
    cvScoresS = model_selection.cross_val_score(s, inputDFS, outputsSerS, cv=10, scoring = s.scorer)
    print(cvScoresS.mean())
    
    dfS, inputCols, outputCol = readData()
    inputCols = ["Ash", "Nonflavanoid Phenols"]
    someCols = inputCols
    dfS = standardize(dfS, someCols)
    
    
    inputDFS = dfS.loc[:, someCols]
    outputsSerS = dfS.loc[:, outputCol]
    
    s = OneNNClassifier()
    cvScoresS = model_selection.cross_val_score(s, inputDFS, outputsSerS, cv=10, scoring = s.scorer)
    print(cvScoresS.mean())


# --------------------------------------------------------------------------------------
# Given
def test07():
    df, inputCols, outputCol = readData()
    
    dfCopy = df.copy()
    standardize(dfCopy, [inputCols[0]])
    print(dfCopy.loc[:, inputCols[0]].head(2))
    
    dfCopy = df.copy()
    normalize(dfCopy, [inputCols[0]])
    print(dfCopy.loc[:, inputCols[0]].head(2))
    
    testSubsets()
