import numpy as np
import pandas as pd
import random
from model.data_analyze import load_and_plot_data, load_and_plot_data1, clean_data

# Format for old data
# time (s)
# Accel X，(m/s^2), Accel Y, Accel Z, Accel total
# It takes about 4 seconds to climb half a floor

Walk_1 = load_and_plot_data("Walking_accel_1.csv", 1, 0, 3, 3)
Walk_2 = load_and_plot_data("Walking_accel_2.csv", 1, 0, 9, 6)
Walk_3 = load_and_plot_data("Walking_accel_3.csv", 1, 0, 6.5, 6)
Stair_1 = load_and_plot_data("Stair_climbing_accel_1.csv", 2, 0, 2, 7)
Stair_2 = load_and_plot_data("Stair_climbing_accel_2.csv", 2, 0, 3, 2.3)
Stair_3 = load_and_plot_data("Stair_climbing_accel_3.csv", 2, 0, 4.5, 3)
Stair_4 = load_and_plot_data("Stair_climbing_accel_4.csv", 2, 0, 1.5, 3.5)
Riding_1 = load_and_plot_data("Riding_accel_1.csv", 3, 0, 5, 12)
Riding_2 = load_and_plot_data("Riding_accel_2.csv", 3, 0, 15, 15)


# Format for new data
# time (s)
# gF X，(FN/Fg), gF Y, gF Z, gF Total
# Accel X，(m/s^2), Accel Y, Accel Z, Accel Total
# gyro X，(rad/s), gyro Y, gyro Z, gyro Total

Walk_4 = load_and_plot_data1("Walking_accel_4.csv", 1, 0, 7, 7)
Walk_5 = load_and_plot_data1("Walking_accel_5.csv", 1, 0, 3, 3)
Walk_6 = load_and_plot_data1("Walking_accel_6.csv", 1, 0, 7.5, 3)
Walk_7 = load_and_plot_data1("Walking_accel_7.csv", 1, 0, 40, 35)
Walk_8 = load_and_plot_data1("Walking_accel_8.csv", 1, 0, 1.5, 4.5)
Walk_9 = load_and_plot_data1("Walking_accel_9.csv", 1, 0, 6, 6)
Stair_5 = load_and_plot_data1("Stair_climbing_accel_5.csv", 2, 0, 5, 3.5)
Stair_6 = load_and_plot_data1("Stair_climbing_accel_6.csv", 2, 0, 4.5, 3)
Stair_7 = load_and_plot_data1("Stair_climbing_accel_7.csv", 2, 0, 3.5, 4.5)
Stair_8 = load_and_plot_data1("Stair_climbing_accel_8.csv", 2, 0, 2.5, 2.5)
Stair_9 = load_and_plot_data1("Stair_climbing_accel_9.csv", 2, 0, 4, 3)
Stair_10 = load_and_plot_data1("Stair_climbing_accel_10.csv", 2, 0, 4, 2)
Stair_11 = load_and_plot_data1("Stair_climbing_accel_11.csv", 2, 0, 3.5, 3.5)
Stair_12 = load_and_plot_data1("Stair_climbing_accel_12.csv", 2, 0, 2.7, 3.5)
Stair_13 = load_and_plot_data1("Stair_climbing_accel_13.csv", 2, 0, 3, 3)
Stair_14 = load_and_plot_data1("Stair_climbing_accel_14.csv", 2, 0, 6, 7)
Stair_15 = load_and_plot_data1("Stair_climbing_accel_15.csv", 2, 0, 2.5, 2.5)
Stair_16 = load_and_plot_data1("Stair_climbing_accel_16.csv", 2, 0, 5, 3)
Riding_3 = load_and_plot_data1("Riding_accel_3.csv", 3, 0, 5, 9)
Riding_4 = load_and_plot_data1("Riding_accel_4.csv", 3, 0, 3.7, 6)
Riding_5 = load_and_plot_data1("Riding_accel_5.csv", 3, 0, 4, 5)
Riding_6 = load_and_plot_data1("Riding_accel_6.csv", 3, 0, 25, 6)
Riding_7 = load_and_plot_data1("Riding_accel_7.csv", 3, 0, 85, 20)
Riding_8 = load_and_plot_data1("Riding_accel_8.csv", 3, 0 ,9.5, 38)


# Old data, Accel X:1, Accel Y:2, Accel Z:3, Accel total:4
variable_type = [1,2,3,4]

# New data, gF X:1, gF Y:2, gF Z:3, gF T:4, Accel X:5, Accel Y:6, Accel Z:7, Accel T:8, gyro X:9, gyro Y:10, gyro Z:11, gyro T:12
variable_type_1 = [1,2,3,4,5,6,7,8,9,10,11,12]

# mean:0, std:1, max:2, min:3, mad:4, skewness:5, kurtosis:6, iqr:7, entropy:8
calculated_type = [0,1,2,3,4,5,6,7]

# Old data
New_Walk_1 = clean_data(Walk_1, variable_type, calculated_type, 1, window_size=2)
New_Walk_2 = clean_data(Walk_2, variable_type, calculated_type, 1, window_size=2)
New_Walk_3 = clean_data(Walk_3, variable_type, calculated_type, 1, window_size=2)
New_Stair_1 = clean_data(Stair_1, variable_type, calculated_type, 2, window_size=2)
New_Stair_2 = clean_data(Stair_2, variable_type, calculated_type, 2, window_size=2)
New_Stair_3 = clean_data(Stair_3, variable_type, calculated_type, 2, window_size=2)
New_Stair_4 = clean_data(Stair_4, variable_type, calculated_type, 2, window_size=2)
New_Riding_1 = clean_data(Riding_1, variable_type, calculated_type, 3, window_size=2)
New_Riding_2 = clean_data(Riding_2, variable_type, calculated_type, 3, window_size=2)

# New data
New_Walk_4 = clean_data(Walk_4, variable_type_1, calculated_type, 1, window_size=2)
New_Walk_5 = clean_data(Walk_5, variable_type_1, calculated_type, 1, window_size=2)
New_Walk_6 = clean_data(Walk_6, variable_type_1, calculated_type, 1, window_size=2)
New_Walk_7 = clean_data(Walk_7, variable_type_1, calculated_type, 1, window_size=2)
New_Walk_8 = clean_data(Walk_8, variable_type_1, calculated_type, 1, window_size=2)
New_Walk_9 = clean_data(Walk_9, variable_type_1, calculated_type, 1, window_size=2)
New_Stair_5 = clean_data(Stair_5, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_6 = clean_data(Stair_6, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_7 = clean_data(Stair_7, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_8 = clean_data(Stair_8, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_9 = clean_data(Stair_9, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_10 = clean_data(Stair_10, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_11 = clean_data(Stair_11, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_12 = clean_data(Stair_12, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_13 = clean_data(Stair_13, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_14 = clean_data(Stair_14, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_15 = clean_data(Stair_15, variable_type_1, calculated_type, 2, window_size=2)
New_Stair_16 = clean_data(Stair_16, variable_type_1, calculated_type, 2, window_size=2)
New_Riding_3 = clean_data(Riding_3, variable_type_1, calculated_type, 3, window_size=2)
New_Riding_4 = clean_data(Riding_4, variable_type_1, calculated_type, 3, window_size=2)
New_Riding_5 = clean_data(Riding_5, variable_type_1, calculated_type, 3, window_size=2)
New_Riding_6 = clean_data(Riding_6, variable_type_1, calculated_type, 3, window_size=2)
New_Riding_7 = clean_data(Riding_7, variable_type_1, calculated_type, 3, window_size=2)
New_Riding_8 = clean_data(Riding_8, variable_type_1, calculated_type, 3, window_size=2)

New_Walk_all = np.vstack((New_Walk_4, New_Walk_5, New_Walk_6, New_Walk_7, New_Walk_8, New_Walk_9))
New_Stair_all = np.vstack((New_Stair_5, New_Stair_6, New_Stair_7, New_Stair_8, New_Stair_9, New_Stair_10, New_Stair_11, New_Stair_12, New_Stair_13, New_Stair_14, New_Stair_15, New_Stair_16))
New_Riding_all = np.vstack((New_Riding_3, New_Riding_4, New_Riding_5, New_Riding_6, New_Riding_7, New_Riding_8))


# Merge data
All_data = np.vstack((New_Walk_all, New_Stair_all, New_Riding_all))
random_index = random.sample(range(All_data.shape[0]), All_data.shape[0])
New_all_data = All_data[random_index, :]
print(New_all_data.shape)

pd.DataFrame(New_all_data, dtype = np.float64).to_csv("Data.csv")

