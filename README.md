# Position Time Series Analysis in Python

ts_dip is a Python script that performs a basic time series analysis using data from permanent GNSS stations

## Input Files

To perform the computations, 'ts_dip.py' needs an imput .cts file. 

.cts files contain the solutions of GNSS stations [Observation Dates, Coordinates, Solution Dates]

## Code

Using the following Ground movement model: 

y = a * x + b + c1 * math.sin(w1 * x) + d1 * math.cos(w1 * x) + c2 * math.sin(w2 * x) + d2 * math.cos(w2 * x) + J1 + .. + Jn

Executes the following steps:
 - Parsing data from cts files
 - Converting geocentric to topocentric coordinates
 - Using Least Squares algorythms to estimate the parameters of the model
 - Detecting outliers and removing them

 ## Outputs

 - Visualises the generated model 
 - Prints the parameters of the estimated model

 ## License

 The work is licensed under [MIT-license](LICENSE)