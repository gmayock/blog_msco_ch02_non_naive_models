import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.3.1_table_2.1', header=None, index_col=0).T
df.reset_index(drop=True, inplace=True)

# # Calculating and showing MA(4)
# This calculates MA(4)
df['MA(4)'] = np.nan
df['MA(4)'][1:] = df['Demand'].rolling(window=4).mean()[:-1]
df['MA(4)_err'] = df['Demand']-df['MA(4)']
print(df)
print("\n Mean Absolute Deviation: \n",abs(df['MA(4)_err']).mean())

# This plots the actual demand and the weekly MA(4) from weeks 5 on
plt.ylabel('Widget Demand (000s)');
plt.ylim(0,15);
plt.axes().yaxis.grid(linestyle=':');
plt.xlabel('Week');
plt.xlim(0,15);
plt.title('Widget Demand Data');
plt.plot(df['Week'], df['Demand'], marker='o');
plt.plot(df['Week'], df['MA(4)'], marker='D');
plt.legend();
plt.show();

# # This calculates the average for the last four weeks of data
m = sum(df['Demand'][-4:])/4
# print(m)

# This creates a new dataframe with the average of the last four weeks of data
df_multi_step = pd.DataFrame([[m, i+13] for i in range(0,9)], columns=['MA(4)', 'Week'])
new_df = pd.merge(df, df_multi_step, on=['MA(4)', 'Week'], how='outer')

# This plots the actual demand and the weekly MA(4) from weeks 5 on (through the longer time period in the new_df)
plt.ylabel('Widget Demand (000s)');
plt.ylim(0,15);
plt.axes().yaxis.grid(linestyle=':');
plt.xlabel('Week');
plt.xlim(0,25);
plt.title('Widget Demand Data');
plt.plot(new_df['Week'], new_df['Demand'], marker='o');
plt.plot(new_df['Week'], new_df['MA(4)'], marker='D');
plt.legend();
plt.show();

# This calculates the exponential smoothing model's forecast based on a first-period forecast equal to first-period demand
alph = 0.1
df['expo_pred'] = df['Demand'].astype('float')
for i in range(1, len(df['expo_pred'])):
    df['expo_pred'][i] = round(alph*df['Demand'][i-1]+(1-alph)*df['expo_pred'][i-1],2)
df['expo_abs_err'] = abs(df['expo_pred']-df['Demand'])
print("\n", df)
print("\n Mean Absolute Deviation: \n", round(df['expo_abs_err'][1:].mean(),2))

# This plots the actual demand and the weekly expo pred from weeks 5 on
plt.ylabel('Widget Demand (000s)');
plt.ylim(0,15);
plt.axes().yaxis.grid(linestyle=':');
plt.xlabel('Week');
plt.xlim(0,15);
plt.title('Widget Demand Data');
plt.plot(df['Week'], df['Demand'], marker='o');
plt.plot(df['Week'][1:], df['expo_pred'][1:], marker='D');
plt.legend();
plt.show();

# # Mimicking the Excel Solver function
df2 = pd.read_excel('filepath/chapter_2_instruction.xlsx', sheet_name='2.3.4_seeking_alpha', header=2, index_col=None, nrows=14)
# print(df2)

def exponentialForecast(period, alpha):
    if period == 1:
        f = 13
    else:
        f = alpha*df2['Dt'][period-2]+(1-alpha)*exponentialForecast(period-1, alpha)
    return f

def exponentialError(period, alpha):
    err = exponentialForecast(period, alpha)-df2['Dt'][period-1]
    return err

alph = .1

print("\n Formulaic forecasts:")
for i in range(1,len(df2['Period'])+1):
    print(exponentialForecast(i, alph))

print("\n The formula's errors:")
for i in range(1,len(df2['Period'])+1):
    print(exponentialError(i, alph))

sq_err = sum([pow(exponentialError(i, alph),2) for i in range(1,len(df2['Period'])+1)])
print("\n Total square error:",sq_err)

df3 = []
for i in range (0, 1000):
    val = [i/1000, sum([pow(exponentialError(q, i/1000),2) for q in range(1,len(df2['Period'])+1)])]
    df3.append(val)
df3 = pd.DataFrame(df3, columns=['alpha_value','squared_error'])

# This plots the sq error dependent upon alpha
plt.ylabel('Squared Error');
plt.ylim(57,59)
plt.axes().yaxis.grid(linestyle=':');
plt.xlabel('Alpha Value');
plt.title('Squared Error Curve by Alpha');
plt.plot(df3['alpha_value'], df3['squared_error']);
plt.legend();
plt.show();

print("\n New: \n",df3)

def totalSquaredError(alpha):
    tot_sq_err = sum([pow(exponentialError(i, alpha),2) for i in range(1,len(df2['Period'])+1)])
    return tot_sq_err

from scipy.optimize import minimize_scalar
glob_min = minimize_scalar(totalSquaredError, bounds=(0, 1), method='Golden')
print("\n",glob_min)