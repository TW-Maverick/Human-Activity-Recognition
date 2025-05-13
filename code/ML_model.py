import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model.data_analyze import load_and_plot_data1, clean_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix


def perform_model(model, model_name:str, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, Walk_1:np.ndarray, Stair_1:np.ndarray, Stair_2:np.ndarray, Riding_1:np.ndarray) -> None: 

    model.fit(X_train, y_train.astype('int'))

    if model_name == "RandomForestClassifier":
        importances = model.feature_importances_

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
    plt.savefig("Results/Model performance/Confusion matrix " + model_name + ".png")
    plt.show()

    # Walk
    X_demo = Walk_1[:,:96]
    y_demo_predict = model.predict(X_demo)
    #y_demo_predict_probability = model.predict_proba(X_demo)
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


    # Climbed stair 1
    X_demo = Stair_1[:,:96]
    y_demo_predict = model.predict(X_demo)
    #y_demo_predict_probability = model.predict_proba(X_demo)
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


    # Climbed stair 2
    X_demo = Stair_2[:,:96]
    y_demo_predict = model.predict(X_demo)
    #y_demo_predict_probability = model.predict_proba(X_demo)
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


    # Riding
    X_demo = Riding_1[:,:96]
    y_demo_predict = model.predict(X_demo)
    #y_demo_predict_probability = model.predict_proba(X_demo)
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


    return importances


# Data for training and testing
Data = pd.read_csv('Data.csv', index_col=0, dtype=np.float64)
print(Data.head())
Data1 = Data.to_numpy()         

# The dataset contains 12 types of observations.
# Each observation is represented by 8 extracted features.
# An additional field is included as the label, indicating the corresponding activity or class.
# The feature sequence follows this structure: gF_X_mean, gF_X_std, gF_X_max, gF_X_min, gF_X_mad, gF_X_skew, gF_X_kurtosis, gF_X_iqr, ...


X_train1 = Data1[:,:96]
y_label = Data1[:,96]

# Split
[X_train, X_test, y_train, y_test] = train_test_split(X_train1, y_label,test_size = 0.3, random_state = 42, stratify = y_label)

# demo
Walk_1 = load_and_plot_data1("Walking_accel_1.csv", 1, 0, 4.5, 4)
Walk_2 = load_and_plot_data1("Walking_accel_2.csv", 1, 0, 4.5, 3)
Stair_1 = load_and_plot_data1("Stair_climbing_accel_1.csv", 2, 0, 3.5, 2.5)
Stair_2 = load_and_plot_data1("Stair_climbing_accel_2.csv", 2, 0, 3.5, 2)
Riding_1 = load_and_plot_data1("Riding_accel_1.csv", 3, 0, 10, 20)

# New data gF X:1, gF Y:2, gF Z:3, gF T:4, Accel X:5, Accel Y:6, Accel Z:7, Accel T:8, gyro X:9, gyro Y:10, gyro Z:11, gyro T:12
variable_type_1 = [1,2,3,4,5,6,7,8,9,10,11,12]
# mean:0, std:1, max:2, min:3, mad:4, skewness:5, kurtosis:6, iqr:7, entropy:8
calculated_type = [0,1,2,3,4,5,6,7]

# Data preprocessing
New_Walk_1 = clean_data(Walk_1, variable_type_1, calculated_type, 1, window_size=2)
New_Walk_2 = clean_data(Walk_2, variable_type_1, calculated_type, 1, window_size=2)
New_Stair_1 = clean_data(Stair_1, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_2 = clean_data(Stair_2, variable_type_1, calculated_type, 2, window_size=2)
New_Riding_1 = clean_data(Riding_1, variable_type_1, calculated_type, 3, window_size=2)


# Build Model
#lr = LogisticRegression( C=1000.0, random_state=0, penalty='l2')
#lr = LogisticRegression( C=1000.0, penalty='l2')
#lr = LogisticRegression(penalty='l2', class_weight = 'balanced', verbose = 1, n_jobs = 1)
#lr = LogisticRegression( C=1000.0, penalty='l2', class_weight = {0:.08, 1:.92})
#lr = LogisticRegression( C=100.0, penalty='l2', random_state = 87 )
#lr = LogisticRegression( C=1000.0, penalty='l1',solver = 'liblinear')
#lr = LogisticRegression(C=1000.0, penalty='l1', solver = 'saga', random_state = 87)

#rfc = RandomForestClassifier()


# start Grid search
'''
parameters = {'C':[0.01, 0.1, 1, 10, 20, 30], 'penalty':['l2','l1']}
lr = LogisticRegression()
lr_grid = GridSearchCV(lr, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
perform_model(lr_grid, "LogisticRegression", X_train, y_train, X_test, y_test, New_Walk_2, New_Stair_1, New_Stair_2, New_Riding_1)

parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
lr_svc = LinearSVC(tol=0.00005)
lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, n_jobs=-1, verbose=1)
perform_model(lr_svc_grid, "LinearSVC", X_train, y_train, X_test, y_test, New_Walk_2, New_Stair_1, New_Stair_2, New_Riding_1)

parameters = {'C':[2,8,16], 'gamma': [ 0.0078125, 0.125, 2]}
rbf_svm = SVC(kernel='rbf')
rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, n_jobs=-1)
perform_model(rbf_svm_grid, "SVC", X_train, y_train, X_test, y_test, New_Walk_2, New_Stair_1, New_Stair_2, New_Riding_1)

parameters = {'max_depth':np.arange(3,10,2)}
dt = DecisionTreeClassifier()
dt_grid = GridSearchCV(dt,param_grid=parameters, n_jobs=-1)
perform_model(dt_grid, "DecisionTreeClassifier", X_train, y_train, X_test, y_test, New_Walk_2, New_Stair_1, New_Stair_2, New_Riding_1)
'''
rfc = RandomForestClassifier()

# Feature Importance Analysis and Dimensionality Reduction
importances = perform_model(rfc, "RandomForestClassifier", X_train, y_train, X_test, y_test, New_Walk_2, New_Stair_1, New_Stair_2, New_Riding_1)
column = np.array(["gF_X_mean", "gF_X_std", "gF_X_max", "gF_X_min", "gF_X_mad", "gF_X_skew", "gF_X_kurtosis", "gF_X_iqr", "gF_Y_mean", "gF_Y_std",
          "gF_Y_max", "gF_Y_min", "gF_Y_mad", "gF_Y_skew", "gF_Y_kurtosis", "gF_Y_iqr", "gF_Z_mean", "gF_Z_std", "gF_Z_max", "gF_Z_min", 
          "gF_Z_mad", "gF_Z_skew", "gF_Z_kurtosis", "gF_Z_iqr", "gF_T_mean", "gF_T_std", "gF_T_max", "gF_T_min", "gF_T_mad", "gF_T_skew",
          "gF_T_kurtosis", "gF_T_iqr", "Accel_X_mean", "Accel_X_std", "Accel_X_max", "Accel_X_min", "Accel_X_mad", "Accel_X_skew", "Accel_X_kurtosis", 
          "Accel_X_iqr", "Accel_Y_mean", "Accel_Y_std", "Accel_Y_max", "Accel_Y_min", "Accel_Y_mad", "Accel_Y_skew", "Accel_Y_kurtosis", "Accel_Y_iqr", 
          "Accel_Z_mean", "Accel_Z_std", "Accel_Z_max", "Accel_Z_min", "Accel_Z_mad", "Accel_Z_skew", "Accel_Z_kurtosis", "Accel_Z_iqr", "Accel_T_mean", 
          "Accel_T_std", "Accel_T_max", "Accel_T_min", "Accel_T_mad", "Accel_T_skew", "Accel_T_kurtosis", "Accel_T_iqr", "Gyro_X_mean", "Gyro_X_std", 
          "Gyro_X_max", "Gyro_X_min", "Gyro_X_mad", "Gyro_X_skew", "Gyro_X_kurtosis", "Gyro_X_iqr", "Gyro_Y_mean", "Gyro_Y_std", "Gyro_Y_max", 
          "Gyro_Y_min", "Gyro_Y_mad", "Gyro_Y_skew", "Gyro_Y_kurtosis", "Gyro_Y_iqr", "Gyro_Z_mean", "Gyro_Z_std", "Gyro_Z_max", "Gyro_Z_min", 
          "Gyro_Z_mad", "Gyro_Z_skew", "Gyro_Z_kurtosis", "Gyro_Z_iqr", "Gyro_T_mean", "Gyro_T_std", "Gyro_T_max", "Gyro_T_min", "Gyro_T_mad", 
          "Gyro_T_skew", "Gyro_T_kurtosis", "Gyro_T_iqr"])

sorted_indices = np.argsort(importances)

# Taking top 20 index 
top_20_indices = sorted_indices[-20:][::-1]

label = column[top_20_indices]
plt.bar(label, height=importances[top_20_indices], tick_label=label)
plt.xticks(rotation=90, ha="right")
plt.title("Feature Importance")
plt.xlabel('feature')
plt.tight_layout()
plt.savefig("Results/Model Performance/Feature Importance.png")
plt.show()


# correlation heat map
ax = sns.heatmap(Data.iloc[:,top_20_indices].corr())
# Set new x and y label
ax.set_xticklabels(label)
ax.set_yticklabels(label)
plt.xticks(rotation=90, ha="right")
plt.yticks(rotation=0)
plt.title('Correlation heatmap', fontsize=15)
plt.tight_layout()
plt.savefig("Results/Model Performance/heatmap.png")
plt.show()