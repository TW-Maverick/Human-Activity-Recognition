## Human-Activity-Recognition (2024H1)
### Advanced Human Activity Recognition (HAR) Using Smartphone Sensors
This research focuses on the field of Human Activity Recognition (HAR), utilizing machine learning (ML) and pattern recognition techniques to automatically detect and classify various human activities. By analyzing data collected from sensors of smartphones, this study aims to identify activities such as walking, climbing stairs, and cycling. We plan to initially use logistic regression models, expecting to achieve an accuracy of at least 95%. Upon success, we will explore more advanced classification algorithms like Linear SVC, Kernal SVC, Decision Tree, and Random Forest Classifier to determine the most suitable model. The outcomes of this study not only validate the feasibility of autonomous data collection and machine learning solutions but also demonstrate their potential applications in areas such as medical diagnosis and treatment.

📁 Dataset Overview  
1. The dataset contains 12 types of observations (e.g., gF X:1, gF Y:2, gF Z:3, gF T:4, Accel X:5, Accel Y:6, Accel Z:7, Accel T:8, gyro X:9, gyro Y:10, gyro Z:11, gyro T:12).  

2. Each observation is represented by 8 extracted features (e.g., mean:0, std:1, max:2, min:3, mad:4, skewness:5, kurtosis:6, iqr:7, entropy:8).  

3. An additional field is included as the label, indicating the corresponding activity or class.

🔗Code Description: 
1. Load_data:
Combine various types of data, e.g., acceleration, angular rate, gravity.

2. data_analyze:
Functions for data collection, cleaning, and visualization.

3. ML_model:
Construction and validation of different machine learning models.

4. Demo_ML_model:
Final classification result display. Simply input data collected from the IMU to identify the type of physical activity, e.g., riding a bicycle, riding a motorcycle, walking.
