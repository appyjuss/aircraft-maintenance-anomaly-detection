# Landing Gear Data Synthetic README

## Overview

The data sets in this collection are shaped (X, Y, Z), where:
- X: Number of datasets
- Y: Length of time series
- Z: Number of parameters

These have been reordered into a 2D array with two new columns:
- 'set_id': Corresponding to X
- 'row_id': Corresponding to Y

## Data Set Details:
-------------------------------------------------------------------------------------

### BogiePitchTrimmer_PressureDrop_OverFlightCycles

**File Name:** 'LGD_S_DATASET_1_BogiePitchTrimmer_PressureDrop_OverFlightCycles.csv' 
**System:** Hydraulics
**Date Created:** 13/11/2023 
**Application:** RUL Estimation (Linear Regression)
**Shape:** (200, 200, 10)  
**Size:** 11.9 MB
**Description:**
This is a synthetic multivariate time series dataset for a Bogie Pitch Trimmer,
a hydraulic system responsible for adjusting the angle of the aircraft's landing 
gear assemblies. The dataset is used to measure the decline in the system's 
hydraulic fluid caused by leakage between flight cycles, across a selection of 
aircraft of the same design. The primary parameters of this dataset are the oil 
pressure in the left and right landing gear, which decreases in line with the 
fluid level. The dataset consists of 200 sets of data, each 200 flight cycles 
long, across 8 different parameters. This data can be used to predict the RUL
of the system, with regression models predicting the decline of the pressure 
until it hits a known critical threshold. Each dataset represents a slight 
general decline, with a lot of noise caused by static affecting the readings
and environmental factors (Ambient temperature, altitude of airport etc.). Some
datasets will contain sharp spikes in pressure where the fluid has been refilled.

**Columns:**
1. 'set_id': Dataset identifier.
2. 'row_id': Row identifier.
3. 'pressure_1' : Oil Pressure for wheel 1.
4. 'pressure_2' : Oil Pressure for wheel 2.
5. 'temperature_1' : Temperature for wheel 1.
6. 'temperature_2' : Temperature  for wheel 2.
7. 'position_1' : Hydraulic position for wheel 1.
8. 'position_2' : Hydraulic position for wheel 2.
9. 'oil_level_1' : Oil level for wheel 1.
10. 'oil_level_2' : Oil level for wheel 2.

-------------------------------------------------------------------------------------

### TyrePressureIndicator_PressureDrop_OverFlightCycles

**File Name:** 'LGD_S_DATASET_2_TyrePressureIndicator_PressureDrop_OverFlightCycles.csv' 
**System:** Tyre Pressure
**Date Created:** 05/12/2023 
**Application:** RUL Estimation (Linear Regression)
**Shape:** (800, 50, 10)  
**Size:** 12 MB
**Description:**
This is a synthetic multivariate time series dataset for a  tyre pressure 
indication system. The datasets consist of pressure readings from 8 sets of tyres
from a set of Airbus aircraft after landing, and track the decline in tyre 
pressure due to leakage. The dataset consists of 800 sets of data, each 50 flight
cycles long, across 8 pressure parameters. Like with the Hydraulics dataset,
regression is used to estimate the RUL of each tyre, so they can be refilled.
These datasets contain a lot of noise from environmental factors, as ambient
temperature and altitude can have a large impact on the pressure in the tyres.
Many of the datasets also contain a spike in pressure somewhere in the data 
where tyres have been refilled

**Columns:**
1. 'set_id': Dataset identifier.
2. 'row_id': Row identifier.
3. 'pressure_1' : Tyre Pressure for wheel 1.
4. 'pressure_2' : Tyre Pressure for wheel 2.
5. 'pressure_3' : Tyre Pressure for wheel 3.
6. 'pressure_4' : Tyre Pressure for wheel 4.
7. 'pressure_5' : Tyre Pressure for wheel 5.
8. 'pressure_6' : Tyre Pressure for wheel 6.
9. 'pressure_7' : Tyre Pressure for wheel 7.
10. 'pressure_8' : Tyre Pressure for wheel 8.

-------------------------------------------------------------------------------------

### TyrePressureIndicator_TemperatureRise_DuringLanding

**File Name:** 'LGD_S_DATASET_3_TyrePressureIndicator_TemperatureRise_DuringLanding.csv' 
**System:** Landing gear brake
**Date Created:** 13/11/2023 
**Application:** Predicting final temperature (Linear regression, ARIMA)
**Shape:** (600, 180, 4)  
**Size:** 10.4 MB
**Description:**
This multivariate dataset records brake temperature and tyre pressure for an 
Airbus aircraft as it lands. By classifying whether the rise and fall in brake
temperature of a set of data is normal or abnormal, worn of failed brakes can 
be identified. The dataset consists of 600 sets of data, each 180 time steps
long with each step representing a minute, with 2 parameters, brake temperature
and tyre pressure. The normality of the data can be classified using 
unsupervised models such as Autoencoders. Each set of data shows the 
brake pressure sharply increases as the aircraft brakes heat up on landing
and slowlydecreases after the peak is reached.

**Columns:**
1. 'set_id': Dataset identifier.
2. 'row_id': Row identifier.
3. 'pressure' : Brake pressure for landing gear wheel.
4. 'temperature' : Brake temperature for landing gear wheel.

-------------------------------------------------------------------------------------

### LandingGearBrakes_MaxTemperature_OverFlightCycles

**File Name:** 'LGD_S_DATASET_4_LandingGearBrakes_MaxTemperature_OverFlightCycles.csv' 
**System:** Brake temperature
**Date Created:** 13/11/2023 
**Application:** Anomaly detection (Clustering, SVM etc.)
**Shape:** (100, 500, 10)  
**Size:** 12.0 MB
**Description:**
This multivariate time series dataset contains the brake temperatures for 8 sets 
of brakes. Anomalies in brake temperature or deceleration may not always indicate
a fault with bearings, but when both are anomalous it indicates the bearings 
are damaged and need replacing. This dataset can be used to train classification
models to differentiate between normal and anomalous readings, however, there 
is very little failure data available due to the rarity of its damaged bearing 
being identified in the field. The dataset consists of 100 sets of data, each 
500 flight cycles long, across 8 parameters for brake temperature. Clustering 
models like the K-Means clustering algorithm can be used to plot the mean brake
temperature to identify anomalous readings. Each set of data appears as a linear
dataset suffering from a lot of noise.

**Columns:**
1. 'set_id': Dataset identifier.
2. 'row_id': Row identifier.
3. 'temperature_1' : Brake temperature for wheel 1.
4. 'temperature_2' : Brake temperature for wheel 2.
5. 'temperature_3' : Brake temperature for wheel 3.
6. 'temperature_4' : Brake temperature for wheel 4.
7. 'temperature_5' : Brake temperature for wheel 5.
8. 'temperature_6' : Brake temperature for wheel 6.
9. 'temperature_7' : Brake temperature for wheel 7.
10. 'temperature_8' : Brake temperature for wheel 8.

-------------------------------------------------------------------------------------


### LandingGearBrakes_Deceleration_DuringTakeoff

**File Name:** 'LGD_S_DATASET_5_LandingGearBrakes_Deceleration_DuringTakeoff.csv' 
**System:** Landing Gear Brakes and Wheels
**Date Created:** 26/11/2023 
**Application:** Anomaly Detection to identify damaged wheel bearings (e.g. LSTM Autoencoder)
**Shape:** (200, 100, 24)  
**Size:** 12.4 MB
**Description:**
This synthetic multivariate dataset records brake temperature, brake pressure
and wheel speed for an Airbus aircraft during the last 100 seconds of take off.
The dataset consists of 200 sets of data, each 100-time steps long with each 
step representing a second during the take off phase. The dataset has 24 
parameters for temperature, pressure, and wheel speed values for each of the 8
wheels on the aircraft, with a total of 24 parameters. Each set of data ends
in a spike of brake temperature as the wheels hit the tarmac, followed by a 
small spike in pressure indicating the brake has been applied which causes the
wheel speed to decrease.

**Columns:**
1. 'set_id': Dataset identifier.
2. 'row_id': Row identifier.
3. 'pressure_1' : Brake pressure for brake 1.
4. 'pressure_2' : Brake pressure for brake 2.
5. 'pressure_3' : Brake pressure for brake 3.
6. 'pressure_4' : Brake pressure for brake 4.
7. 'pressure_5' : Brake pressure for brake 5.
8. 'pressure_6' : Brake pressure for brake 6.
9. 'pressure_7' : Brake pressure for brake 7.
10. 'pressure_8' : Brake pressure for brake 8.
11. 'temperature_1' : Temperature for brake 1.
12. 'temperature_2' : Temperature for brake 2.
13. 'temperature_3' : Temperature for brake 3.
14. 'temperature_4' : Temperature for brake 4.
15. 'temperature_5' : Temperature for brake 5.
16. 'temperature_6' : Temperature for brake 6.
17. 'temperature_7' : Temperature for brake 7.
18. 'temperature_8' : Temperature for brake 8.
19. 'wheel_speed_1' : Wheel speed for wheel 1.
20. 'wheel_speed_2' : Wheel speed for wheel 2.
21. 'wheel_speed_3' : Wheel speed for wheel 3.
22. 'wheel_speed_4' : Wheel speed for wheel 4.
23. 'wheel_speed_5' : Wheel speed for wheel 5.
24. 'wheel_speed_6' : Wheel speed for wheel 6.
25. 'wheel_speed_7' : Wheel speed for wheel 7.
26. 'wheel_speed_8' : Wheel speed for wheel 8.

-------------------------------------------------------------------------------------


### LandingGearBrakes_Deceleration_OverFlightCycles

**File Name:** 'LGD_S_DATASET_6_LandingGearBrakes_Deceleration_OverFlightCycles.csv' 
**System:** Landing Gear Brakes and Wheels
**Date Created:** 26/11/2023 
**Application:** Anomaly Detection to identify damaged wheel bearings (e.g. LSTM Autoencoder)
**Shape:** (400, 50, 24)  
**Size:** 12.5 MB
**Description:**
This is a synthetic multivariate time series dataset for a Bogie Pitch Trimmer,
This synthetic multivariate dataset records the max brake temperature, max brake
pressure, and deceleration for each wheel on an Airbus aircraft during takeoff.
The combination of max brake temperature and deceleration can be analysed to
identify anomalise using clustering of support vector machines to identify faults
with wheel bearings. This dataset is essentially the calculated values of the
LGD_S_DATASET_5, where the max values for each parameter were identified, and the
deceleration at the point the brake was applied was calculated.

**Columns:**
1. 'set_id': Dataset identifier.
2. 'row_id': Row identifier.
3. 'max_pressure_1' : Brake pressure for brake 1.
4. 'max_pressure_2' : Brake pressure for brake 2.
5. 'max_pressure_3' : Brake pressure for brake 3.
6. 'max_pressure_4' : Brake pressure for brake 4.
7. 'max_pressure_5' : Brake pressure for brake 5.
8. 'max_pressure_6' : Brake pressure for brake 6.
9. 'max_pressure_7' : Brake pressure for brake 7.
10. 'max_pressure_8' : Brake pressure for brake 8.
11. 'max_temperature_1' : Temperature for brake 1.
12. 'max_temperature_2' : Temperature for brake 2.
13. 'max_temperature_3' : Temperature for brake 3.
14. 'max_temperature_4' : Temperature for brake 4.
15. 'max_temperature_5' : Temperature for brake 5.
16. 'max_temperature_6' : Temperature for brake 6.
17. 'max_temperature_7' : Temperature for brake 7.
18. 'max_temperature_8' : Temperature for brake 8.
19. 'deceleration_1' : Deceleration for wheel 1.
20. 'deceleration_2' : Deceleration for wheel 2.
21. 'deceleration_3' : Deceleration for wheel 3.
22. 'deceleration_4' : Deceleration for wheel 4.
23. 'deceleration_5' : Deceleration for wheel 5.
24. 'deceleration_6' : Deceleration for wheel 6.
25. 'deceleration_7' : Deceleration for wheel 7.
26. 'deceleration_8' : Deceleration for wheel 8.
