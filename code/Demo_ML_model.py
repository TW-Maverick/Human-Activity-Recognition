import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.data_analyze import load_and_plot_data1, clean_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from prettytable import PrettyTable



def perform_model(model, model_name:str, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, Activity_type:np.ndarray, Activity_type_1:str) -> None: 

    model.fit(X_train, y_train.astype('int'))

    if model_name == "RandomForestClassifier":
        importances = model.feature_importances_
    else:
        importances = 0

    y_predict = model.predict(X_test)

    y_predict = y_predict.reshape(y_predict.shape[0],1)
    y_test = y_test.reshape(y_predict.shape[0],1)
    pa = np.hstack([y_predict, y_test])
    ap = pd.DataFrame(pa, columns=['Predict','Actual'])
    print(ap.head())


    # Model Performance
    labels=['Walking', 'Climbed stair','Riding']
    print('Accuracy : %0.4f' %  accuracy_score(y_test.astype('int'), y_predict.astype('int')))
    print('R2-score : %0.4f' % r2_score(y_test.astype('int'), y_predict.astype('int')))
    print('Mean_squared_error : %0.4f' % mean_squared_error(y_test.astype('int'), y_predict.astype('int')))

    # classification_report
    print(classification_report(y_test.astype('int'), y_predict.astype('int'), target_names=labels))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp = ConfusionMatrixDisplay(confusion_matrix(y_test.astype('int'), y_predict.astype('int')),display_labels=labels,)
    cmp.plot(ax=ax)
    plt.title('Confusion matrix of ' + model_name)
    plt.savefig("Results/Model Performance/Confusion matrix " + model_name + " " + Activity_type_1 + ".png")
    plt.show()


    # Activity_type
    X_demo = Activity_type[:,:96]
    y_demo_predict = model.predict(X_demo)
    y_demo_predict = y_demo_predict.reshape(y_demo_predict.shape[0],1)

    print(y_demo_predict.shape)
    print(y_demo_predict[:10])

    walking_num = np.array(np.where(y_demo_predict == 1))
    print('Walking : ',walking_num.shape[1])
    print('Probability : %0.4f' % float(walking_num.shape[1]/y_demo_predict.shape[0]))
    print("-"*100)
    stair_num = np.array(np.where(y_demo_predict == 2))
    print('Climbed stair : ',stair_num.shape[1])
    print('Probability : %0.4f' % float(stair_num.shape[1]/y_demo_predict.shape[0]))
    print("-"*100)
    riding_num = np.array(np.where(y_demo_predict == 3))
    print('Riding : ',riding_num.shape[1])
    print('Probability : %0.4f' % float(riding_num.shape[1]/y_demo_predict.shape[0]))

    return importances, accuracy_score(y_test.astype('int'), y_predict.astype('int'))


# Data for training and testing
Data = pd.read_csv('Data.csv', index_col=0, dtype=np.float64)
Data1 = Data.to_numpy()         

X_train1 = Data1[:,:96]
y_label = Data1[:,96]

# Split
[X_train, X_test, y_train, y_test] = train_test_split(X_train1, y_label,test_size = 0.3, random_state = 42, stratify = y_label)

# demo
Walk_1 = load_and_plot_data1("Walking_accel_1.csv", 1, 0, 4.5, 4)

# New data gF X:1, gF Y:2, gF Z:3, gF T:4, Accel X:5, Accel Y:6, Accel Z:7, Accel T:8, gyro X:9, gyro Y:10, gyro Z:11, gyro T:12
variable_type_1 = [1,2,3,4,5,6,7,8,9,10,11,12]
# mean:0, std:1, max:2, min:3, mad:4, skewness(偏度):5, kurtosis(峰度):6, iqr(四分位距):7, entropy(熵):8
calculated_type = [0,1,2,3,4,5,6,7]

# Data preprocessing
New_Walk_1 = clean_data(Walk_1, variable_type_1, calculated_type, 1, window_size=2)


# Build a model
# start Grid search

parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
lr = LogisticRegression()
lr_grid = GridSearchCV(lr, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
_, accuracy_lr = perform_model(lr_grid, "LogisticRegression", X_train, y_train, X_test, y_test, New_Walk_1, "Walk_1")

parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
lr_svc = LinearSVC(tol=0.00005)
lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
_, accuracy_lr_svc = perform_model(lr_svc_grid, "LinearSVC", X_train, y_train, X_test, y_test, New_Walk_1, "Walk_1")

parameters = {'C':[2,8,16], 'gamma': [ 0.0078125, 0.125, 2]}
rbf_svm = SVC(kernel='rbf')
rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
_, accuracy_rbf_svm = perform_model(rbf_svm_grid, "SVC", X_train, y_train, X_test, y_test, New_Walk_1, "Walk_1")

parameters = {'max_depth':np.arange(3,10,2)}
dt = DecisionTreeClassifier()
dt_grid = GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
_, accuracy_dt = perform_model(dt_grid, "DecisionTreeClassifier", X_train, y_train, X_test, y_test, New_Walk_1, "Walk_1")

rfc = RandomForestClassifier()
# Feature Importance Analysis and Dimensionality Reduction
importances, accuracy_rfc = perform_model(rfc, "RandomForestClassifier", X_train, y_train, X_test, y_test, New_Walk_1, "Walk_1")


# Accuracy of different models
table = PrettyTable()
table.field_names = ["Model","Hyperparameter Tuing","Accuracy"]
table.add_row(["Logistic Regression","GridSearchCV", np.round(accuracy_lr, 4)])
table.add_row(["Linear SVC","GridSearchCV", np.round(accuracy_lr_svc, 4)])
table.add_row(["Kernal SVM","GridSearchCV", np.round(accuracy_rbf_svm, 4)])
table.add_row(["Decision Tree","GridSearchCV", np.round(accuracy_dt, 4)])
table.add_row(["Random Forest Classifier","None", np.round(accuracy_rfc, 4)])

print(table.get_string(title="ML Model Results"))
