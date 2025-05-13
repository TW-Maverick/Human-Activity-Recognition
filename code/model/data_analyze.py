import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation, skew, kurtosis, entropy, iqr


def load_and_plot_data1(filename:str, label:int, whether_plot:int, strat:float = 0, end:float = 0) -> np.ndarray:

    # label -> 1:walking  2:stair  3:riding

    # strat, end -> time(sec)

    # load data
    # Column 1：time (s)
    # Column 2：gF X，(FN/Fg)
    # Column 3：gF Y
    # Column 4：gF Z
    # Column 5：Accel X，(m/s^2)
    # Column 6：Accel Y
    # Column 7：Accel Z
    # Column 8：gyro X，(rad/s)
    # Column 9：gyro Y
    # Column 10：gyro Z
    absolute_path = 'data/Acceleration (New) from IMU/' + filename
    with open(absolute_path, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)-2                     

        time = []
        gF_x = []
        gF_y = []
        gF_z = []
        gF_T = []
        accel_x = []
        accel_y = []
        accel_z = []
        accel_T = []
        gyro_x = []
        gyro_y = []
        gyro_z = []
        gyro_T = []

        for i in range(line_count):
            Data_stream = lines[i+2].split(',')

            time.append(float(Data_stream[0]))
            gF_x.append(float(Data_stream[1]))
            gF_y.append(float(Data_stream[2]))
            gF_z.append(float(Data_stream[3]))
            gF_T.append(np.sqrt(float(Data_stream[1])**2 + float(Data_stream[2])**2 + float(Data_stream[3])**2))
            accel_x.append(float(Data_stream[4]))
            accel_y.append(float(Data_stream[5]))
            accel_z.append(float(Data_stream[6]))
            accel_T.append(np.sqrt(float(Data_stream[4])**2 + float(Data_stream[5])**2 + float(Data_stream[6])**2))
            gyro_x.append(float(Data_stream[7]))
            gyro_y.append(float(Data_stream[8]))
            gyro_z.append(float(Data_stream[9]))
            gyro_T.append(np.sqrt(float(Data_stream[7])**2 + float(Data_stream[8])**2 + float(Data_stream[9])**2))


    #Data = tuple([time]) + tuple([gF_x]) + tuple([gF_y]) + tuple([gF_z]) + tuple([accel_x]) + \
    #     tuple([accel_y]) + tuple([accel_z]) + tuple([gyro_x]) + tuple([gyro_y]) + tuple([gyro_z])
            
    Data = np.array([time, gF_x, gF_y, gF_z, gF_T, accel_x, accel_y, accel_z, accel_T, gyro_x, gyro_y, gyro_z, gyro_T]).T

    # Eliminate specific seconds
    if (strat and end) == 0:
        label_1ist = np.full((line_count, 1), label)
        Data = np.hstack((Data, label_1ist))

    else:
        start_idx = np.argmin(abs((Data[:,0] - Data[0,0]) - strat))
        end_idx = np.argmin(abs((Data[line_count-1,0] - Data[:,0]) - end))

        label_1ist = np.full(((end_idx-start_idx), 1), label)
        Data = Data[start_idx:end_idx, :]
        Data = np.hstack((Data, label_1ist))


    if whether_plot == 1:
    
        fig, ax = plt.subplots(3,1)
        plt.subplot(3,1,1)
        plt.plot(Data[:,0], Data[:,1], linewidth = 2)                                                # gF_x
        plt.ylabel('gForce_x  (m/s^2)')
        plt.xticks([])
        plt.title(filename[:-12])

        plt.subplot(3,1,2)
        plt.plot(Data[:,0], Data[:,2], linewidth = 2)                                                # gF_y
        plt.ylabel('gForce_y  (m/s^2)')
        plt.xticks([])

        plt.subplot(3,1,3)
        plt.plot(Data[:,0], Data[:,3], linewidth = 2)                                                # gF_z
        ticks = [min(Data[:,0]), max(Data[:,0])]
        plt.xticks(ticks)
        plt.xlabel('time (sec)')
        plt.ylabel('gForce_z  (m/s^2)')
        plt.show()
        fig.savefig("Results/New data from IMU/" + filename[:-12] + filename[-5] + "_gForce" + '.png')


        fig, ax = plt.subplots(3,1)
        plt.subplot(3,1,1)
        plt.plot(Data[:,0], Data[:,5], linewidth = 2)                                                # accel_x
        plt.ylabel('Accel_x  (m/s^2)')
        plt.xticks([])
        plt.title(filename[:-12])

        plt.subplot(3,1,2)
        plt.plot(Data[:,0], Data[:,6], linewidth = 2)                                                # accel_y
        plt.ylabel('Accel_y  (m/s^2)')
        plt.xticks([])

        plt.subplot(3,1,3)
        plt.plot(Data[:,0], Data[:,7], linewidth = 2)                                                # accel_z
        ticks = [min(Data[:,0]), max(Data[:,0])]
        plt.xticks(ticks)
        plt.xlabel('time (sec)')
        plt.ylabel('Accel_z  (m/s^2)')
        plt.show()
        fig.savefig("Results/New data from IMU/" + filename[:-12] + filename[-5] + "_accel" + '.png')


        fig, ax = plt.subplots(3,1)
        plt.subplot(3,1,1)
        plt.plot(Data[:,0], Data[:,9], linewidth = 2)                                                # gyro_x
        plt.ylabel('gyro_x  (rad/s)')
        plt.xticks([])
        plt.title(filename[:-12])

        plt.subplot(3,1,2)
        plt.plot(Data[:,0], Data[:,10], linewidth = 2)                                                # gyro_y
        plt.ylabel('gyro_y  (rad/s)')
        plt.xticks([])

        plt.subplot(3,1,3)
        plt.plot(Data[:,0], Data[:,11], linewidth = 2)                                                # gyro_z
        ticks = [min(Data[:,0]), max(Data[:,0])]
        plt.xticks(ticks)
        plt.xlabel('time (sec)')
        plt.ylabel('gyro_z  (rad/s)')
        plt.show()
        fig.savefig("Results/New data from IMU/" + filename[:-12] + filename[-5] + "_gyro" + '.png')


        fig, ax = plt.subplots(3,1)
        plt.subplot(3,1,1)
        plt.plot(Data[:,0], Data[:,4], linewidth = 2)                                                # gyro_x
        plt.ylabel('gForce_T  (m/s^2)')
        plt.xticks([])
        plt.title(filename[:-12])

        plt.subplot(3,1,2)
        plt.plot(Data[:,0], Data[:,8], linewidth = 2)                                                # gyro_y
        plt.ylabel('Accel_T  (m/s^2)')
        plt.xticks([])

        plt.subplot(3,1,3)
        plt.plot(Data[:,0], Data[:,12], linewidth = 2)                                                # gyro_z
        ticks = [min(Data[:,0]), max(Data[:,0])]
        plt.xticks(ticks)
        plt.xlabel('time (sec)')
        plt.ylabel('gyro_z  (rad/s)')
        plt.show()
        fig.savefig("Results/New data from IMU/" + filename[:-12] + filename[-5] + "_total" + '.png')


    return Data


def load_and_plot_data(filename:str, label:int, whether_plot:int, strat:float = 0, end:float = 0) -> np.ndarray:
    # load data
    # Column 1：time (s)
    # Column 2：Accel X
    # Column 3：Accel Y
    # Column 4：Accel Z
    # Column 5：Accel total
    absolute_path = 'data/Acceleration (New) from IMU/' + filename
    with open(absolute_path, 'r') as file:
        lines = file.readlines()
        line_count = len(lines)-1                     

        time = []
        accel_x = []
        accel_y = []
        accel_z = []
        accel_total = []
        label_1ist = np.full((line_count, 1), label)

        for i in range(line_count):
            Data_stream = lines[i+1].split(',')

            time.append(float(Data_stream[0]))
            accel_x.append(float(Data_stream[1]))
            accel_y.append(float(Data_stream[2]))
            accel_z.append(float(Data_stream[3]))
            accel_total.append(float(Data_stream[4]))


    Data = np.array([time, accel_x, accel_y, accel_z, accel_total]).T

    if (strat and end) == 0:
        label_1ist = np.full((line_count, 1), label)
        Data = np.hstack((Data, label_1ist))

    else:
        start_idx = np.argmin(abs((Data[:,0] - Data[0,0]) - strat))
        end_idx = np.argmin(abs((Data[line_count-1,0] - Data[:,0]) - end))

        label_1ist = np.full(((end_idx-start_idx), 1), label)
        Data = Data[start_idx:end_idx, :]
        Data = np.hstack((Data, label_1ist))
    

    if whether_plot == 1:

        fig, ax = plt.subplots(3,1)
        plt.subplot(3,1,1)
        plt.plot(Data[:,0], Data[:,1], linewidth = 2)                                                # accel_x
        plt.ylabel('Accel_x  (m/s^2)')
        plt.xticks([])
        plt.title(filename[:-12])

        plt.subplot(3,1,2)
        plt.plot(Data[:,0], Data[:,2], linewidth = 2)                                                # accel_y
        plt.ylabel('Accel_y  (m/s^2)')
        plt.xticks([])

        plt.subplot(3,1,3)
        plt.plot(Data[:,0], Data[:,3], linewidth = 2)                                                # accel_z
        ticks = [min(Data[:,0]), max(Data[:,0])]
        plt.xticks(ticks)
        plt.xlabel('time (sec)')
        plt.ylabel('Accel_z  (m/s^2)')
        plt.show()
        fig.savefig("Results/New data from IMU/" + filename[:-12] + filename[-5] + "_gyro" + '.png')


    return Data


def calculate_win_num(Data:np.ndarray, window_size:float, overlapping:float) -> int:
    Data_length = Data.shape[0]

    # number a window of 2.5 seconds (default)
    win_idx = np.argmin(abs((Data[:,0] - Data[0,0]) - window_size))

    # overlapping 50% (default)
    win_num = Data_length//(int(win_idx*(1-overlapping/100)))

    return win_idx, win_num, Data_length, Data.shape[1]


def data_generate(Data:np.ndarray, variable_type:int, calculated_type:int, window_size:float, overlapping:float) -> np.ndarray:

    win_idx, win_num, Data_length, _ = calculate_win_num(Data, window_size, overlapping)

    generate = np.ones([win_num, ])
    for i in range(win_num):
        start_idx = 0 + int(win_idx*(1-overlapping/100))*i
        end_idx = win_idx + int(win_idx*(1-overlapping/100))*i
        if end_idx > Data_length:
            end_idx = Data_length

        if calculated_type == 0:
            generate[i, ] = np.mean(Data[start_idx:end_idx,variable_type])
        elif calculated_type == 1:
            generate[i, ] = np.std(Data[start_idx:end_idx,variable_type])
        elif calculated_type == 2:
            generate[i, ] = np.amax(Data[start_idx:end_idx,variable_type])
        elif calculated_type == 3:
            generate[i, ] = np.amin(Data[start_idx:end_idx,variable_type])
        elif calculated_type == 4:
            generate[i, ] = median_abs_deviation(Data[start_idx:end_idx,variable_type])
        elif calculated_type == 5:
            generate[i, ] = skew(Data[start_idx:end_idx,variable_type])
        elif calculated_type == 6:
            generate[i, ] = kurtosis(Data[start_idx:end_idx,variable_type])
        elif calculated_type == 7:
            generate[i, ] = iqr(Data[start_idx:end_idx,variable_type])   

    return generate


def clean_data(Data:np.ndarray, variable_type:list, calculated_type:list, label:int, window_size:float = 2.5, overlapping:float = 50.0) -> np.ndarray:

    _, win_num, _, _ = calculate_win_num(Data, window_size, overlapping)
    variable_type_length = len(variable_type)
    calculated_type_length = len(calculated_type)

    cleaned_data = np.ones([win_num, (variable_type_length*calculated_type_length)])

    for i in range(variable_type_length):
        for j in range(calculated_type_length):
            cleaned_data[:, i*calculated_type_length+j] = data_generate(Data, variable_type[i], calculated_type[j], window_size, overlapping)

    label_1ist = np.full((win_num, 1), label)
    cleaned_data = np.hstack((cleaned_data, label_1ist))

    return cleaned_data
