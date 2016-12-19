""" Writing our first randomforest code.
Author : NaHopayega
Date : 18th dec 2016

""" 
import pandas as pd
import numpy as np
import random
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model, decomposition, datasets
from collections import Counter
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# defining globs
Ports_dict = {}
ids = None

def prepareTrainingData():
    global Ports_dict, ids
    # Data cleanup
    # TRAIN DATA
    train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.

    # female = 0, Male = 1
    train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    #removing least important class and a dependent class Pclass(dependent on Fare)
    #train_df.drop(['Fare'], axis=1, inplace=True)



    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

    # All missing Embarked -> just make them embark from most common place
    if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
        train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values
    #train_df['Embarked'].fillna(lambda x: random.choice(train_df[train_df['Embarked'] != np.nan]['Embarked']), inplace =True)


    Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
    Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
    train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

    # All the ages with no data -> make the mean of all Ages
    mean_age = train_df['Age'].dropna().mean()
    if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
        train_df.loc[ (train_df.Age.isnull()), 'Age'] = mean_age
    # Making categories of the Age class
    '''
    for i, row in df.iterrows():
        ifor_val = 
        if():
            ifor_val = something_else
            df.set_value(i,'Age',ifor_val)
    '''     

    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    #removing feature fare from train_df
    train_df.drop(['Fare'], axis=1, inplace=True)
    train_data = train_df.values
    # Forming categories on the basis of Age in train_data
    for x in range(0,891):
        if(train_data[x,2]>0 and train_data[x,2]<5):
            train_data[x,2]=1
        if(train_data[x,2]>=5 and train_data[x,2]<10):
            train_data[x,2]=2
        if(train_data[x,2]>=10 and train_data[x,2]<15):
            train_data[x,2]=3
        if(train_data[x,2]>=15 and train_data[x,2]<60):
            train_data[x,2]=4
        if(train_data[x,2]>=60):
            train_data[x,2]=5
        
    return train_data[:, 1:], train_data[:, 0]

def prepareTestData():
    global Ports_dict, ids
    # TEST DATA
    test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

    # I need to do the same with the test data now, so that the columns are the same as the training data
    # I need to convert all strings to integer classifiers:
    # female = 0, Male = 1
    test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Embarked from 'C', 'Q', 'S'
    # All missing Embarked -> just make them embark from most common place
    if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
        test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
    # Again convert all Embarked strings to int
    test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


    # All the ages with no data -> make the median of all Ages
    mean_age = test_df['Age'].dropna().mean()
    if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
        test_df.loc[ (test_df.Age.isnull()), 'Age'] = mean_age

    # All the missing Fares -> assume median of their respective class
    if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
        mean_fare = np.zeros(3)
        for f in range(0,3):                                              # loop 0 to 2
            mean_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().mean()
        for f in range(0,3):                                              # loop 0 to 2
            test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = mean_fare[f]

    # Collect the test data's PassengerIds before dropping it
    ids = test_df['PassengerId'].values
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 



    #removing feature fare from test df
    test_df.drop(['Fare'], axis=1, inplace=True)

    # The data is now ready to go. So lets fit to the train, then predict to the test!
    # Convert back to a numpy array
    test_data = test_df.values
    # Forming categories on the basis of Age in test_data
    for x in range(0,418):
        if(test_data[x,1]>0 and test_data[x,1]<5):
            test_data[x,1]=1
        if(test_data[x,1]>=5 and test_data[x,1]<10):
            test_data[x,1]=2
        if(test_data[x,1]>=10 and test_data[x,1]<15):
            test_data[x,1]=3
        if(test_data[x,1]>=15 and test_data[x,1]<60):
            test_data[x,1]=4
        if(test_data[x,1]>=60):
            test_data[x,1]=5

    return test_data


def testWithAlgo(algo_str, X_train, y_train, X_test):
    print 'Training...'
    if algo_str is "rforest":
        forest = RandomForestClassifier(n_estimators=140,criterion='entropy',min_samples_split=5)
        forest = forest.fit( X_train, y_train )
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(0,5):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        clf=forest
    elif algo_str is "logistic":
        logreg = linear_model.LogisticRegression(tol=1e-5,C=1e4,solver='newton-cg',max_iter=500)
        logreg = logreg.fit(X_train, y_train)
        clf = logreg

    elif algo_str is "svm":
        #clf = SVC()
        #clf=clf.fit(train_data[0::,1::], train_data[0::,0])
        clf=LinearSVC(C=1e3,dual=False,random_state=42)
        clf=clf.fit(X_train, y_train)
        #print('Original dataset shape  smote {}'.format(Counter(train_data[0::,0])))
        #print('Original dataset shape  smote {}'.format(Counter(train_data[0::,5])))

    print 'Predicting...'
    output = clf.predict(X_test).astype(int)


    predictions_file = open("myfirstforest.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(ids, output))
    predictions_file.close()
    print 'Done.'

if __name__ == '__main__':
    X_train, y_train = prepareTrainingData()
    X_test = prepareTestData()
    testWithAlgo('logistic', X_train, y_train, X_test)
