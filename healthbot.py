import re
import csv
import pandas as pd
import pyttsx3
from sklearn import preprocessing
# DecisionTreeClassifier is a class for constructing decision tree classifiers. _tree is an internal module that contains the data structure for decision trees.
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
# imports the cross_val_score function from the model_selection module in sklearn. This function is used to perform cross-validation, which is a technique for assessing the performance of a machine learning model
from sklearn.model_selection import cross_val_score
# SVC is a class for implementing Support Vector Machine (SVM) classifiers.
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Data/training.csv')
testing = pd.read_csv('Data/testing.csv')
cols = training.columns
cols = cols[:-1]  # selects all columns up to the last one but excludes it
x = training[cols]
y = training['diagnosis']
y1 = y

reduced_data = training.groupby(training['diagnosis']).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.50, random_state=42)
# x_train: the feature matrix for the training set
# x_test: the feature matrix for the test set
# y_train: the target variable vector for the training set
# y_test: the target variable vector for the test set

testx = testing[cols]
testy = testing['diagnosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print(scores.mean())

model = SVC()
# trains the SVM model using the training data x_train (which contains the features) and y_train (which contains the labels).
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


def readn(nstr):
    engine = pyttsx3.init()  # creates an instance of the pyttsx3.Engine class.
    # sets the voice of the engine to English language with a female voice
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)  # sets voice speed to 130wpm
    engine.say(nstr)
    # start the speech engine and wait for it to finish speaking the input string before the function terminates.
    engine.runAndWait()
    engine.stop()


severityDictionary = dict()
description_list = dict()
preventionDictionary = dict()
symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1) > 13):
        print("You should visit the nearest hospital and acquire professional medical assistance")
    else:
        print("It might not be that bad but you should take precautionary measures")


def getDescription():
    global description_list
    with open('SuperData/Disease_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('SuperData/symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getpreventionDict():
    global preventionDictionary
    with open('SuperData/disease_prevention.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prev = {row[0]: [row[1], row[2], row[3], row[4]]}
            preventionDictionary.update(_prev)


def getInfo():
    print("-----------------------------------HealthBot-----------------------------------")
    print("\nGreetings!!! What is your name? \t\t\t\t", end="->")
    name = input("")
    print("Hello", name, ". I am a Healthbot")


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if(len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []

# This function takes a list of diseases dis_list and an input string inp.
# It checks if the input string matches any of the diseases in the list using regular expressions.
# The input string is first modified by replacing spaces with underscores, and then a regular expression pattern is created using the modified input string.
# The search method of the re module is then used to search for the pattern in each of the diseases in the list. If a match is found, the disease is added to a pred_list.
# If no matches are found, an empty list is returned.
# The function returns a tuple containing a binary value indicating whether or not a match was found (1 for match, 0 for no match) and the list of predicted diseases.


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/training.csv')
    X = df.iloc[:, :-1]
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.50, random_state=40)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1
    return rf_clf.predict([input_vector])

# Reads the training dataset Training.csv using pandas and splits it into training and testing sets.
# Creates a decision tree classifier object rf_clf and fits the model to the training set using X_train and y_train.
# Creates a dictionary that maps each symptom to its index in the X dataframe.
# Creates an input vector of zeros with the same length as the X dataframe.
# For each symptom in symptoms_exp, sets the corresponding value in the input vector to 1.
# Predicts the disease using the decision tree classifier and the input vector.
# Returns the predicted disease.


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i !=
                    _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nName one symptom you are experiencing \t\t", end="->")
        symptom_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, symptom_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            symptom_input = cnf_dis[conf_inp]
            break
            print("Did you mean: ", cnf_dis, "?(yes/no) :", end="")
            conf_inp = input("")
            if(conf_inp == "yes"):
                break
        else:
            print("Please enter a valid symptom. If you are seeing this message repeatedly it means I have not been trained on the symptoms you're giving and I aplogize for any inconvenience")

    while True:
        try:
            num_days = int(input("Okay. For how many days ? : "))
            break
        except:
            print("Intger values only please.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == symptom_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero(
            )]
            dis_list = list(symptoms_present)
            if len(dis_list) != 0:
                print("symptoms present  " + str(list(symptoms_present)))
            print("symptoms given " + str(list(symptoms_given)))
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if(inp == "yes" or inp == "no"):
                        break
                    else:
                        print(
                            "Please provide proper answers i.e. (yes/no) : ", end="")
                if(inp == "yes"):
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp, num_days)
            if(present_disease[0] == second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ",
                      present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            prevention_list = preventionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(prevention_list):
                print(i+1, ")", j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)


getSeverityDict()
getDescription()
getpreventionDict()
getInfo()
tree_to_code(clf, cols)
print("----------------------------------------------------------------------------------------")
