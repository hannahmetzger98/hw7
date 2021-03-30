import pandas as pd
import scipy.spatial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
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
    
    #Orignal dataset
    dfOg = df.copy()
    dfOg = accuracyOfActualVsPredicted(outputCol, inputCols)
    print(dfOg)
    
    #norm dataset
    dfN = df.copy()
    dfN = testNormalize()
    print(dfN)
    
    #stan dataset
    dfS = df.copy()
    dfS = testStandardize()
    print(dfN)
    
    


comparePreprocessing()






# -----------------------------------------------------------------------------
# Problem 4




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
