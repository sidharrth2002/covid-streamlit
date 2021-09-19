import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns

def get_result(wave,wave_num,st) :
    population = pd.read_csv('./data/population.csv')
    population.drop(['pop_18','pop_60','idxs'],axis='columns', inplace=True)
    population.drop(0,axis='rows', inplace=True)
    state_index = np.arange(16)
    population.index = state_index
    temp = population.iloc[6]
    population.iloc[6] = population.iloc[7]
    population.iloc[7] = temp
    temp = population.iloc[7]
    population.iloc[7] = population.iloc[8]
    population.iloc[8] = temp
    temp = population.iloc[9]
    population.iloc[9] = population.iloc[11]
    population.iloc[11] = temp
    temp = population.iloc[10]
    population.iloc[10] = population.iloc[12]
    population.iloc[12] = temp
    #calculate the total cases for each states
    print('Wave '+ str(wave_num) +' : each states total cases')
    wave = wave.groupby(["state"]).agg({"cases_new": "sum"})
    wave = wave.reset_index()
    wave = wave.rename(columns={"cases_new" : "total_cases"})
    wave["total_cases"] = wave["total_cases"].astype('float')
    
    #calculate the total cases / each state population for each state
    each_state_total_cases = wave
    each_state_total_cases["total_cases"] = each_state_total_cases["total_cases"] / population["pop"]
    each_state_total_cases_devide_by_population = each_state_total_cases
    each_state_total_cases_devide_by_population = each_state_total_cases_devide_by_population.rename(columns={"total_cases" : "total_cases / Population"})
    
    #normalize the datasets
    column_maxes = each_state_total_cases_devide_by_population['total_cases / Population'].max()
    each_state_total_cases_devide_by_population['total_cases / Population'] = each_state_total_cases_devide_by_population['total_cases / Population'] / column_maxes
    print('Wave '+ str(wave_num) +' : each states total cases / polution after normalize')
    
    outlierDetection = each_state_total_cases_devide_by_population
    
    #display the boxplot of each state total_cases / Population

    fig, ax = plt.subplots(figsize=(10, 2))
    sns.boxplot(x=outlierDetection['total_cases / Population'], ax=ax)
    ax.set_title('Wave '+ str(wave_num))
    st.pyplot(fig)

    #to calculate the IQR so we can find for the outlier
    Q1 = outlierDetection.quantile(0.25)
    Q3 = outlierDetection.quantile(0.75)
    IQR = Q3 - Q1
    IQR['total_cases / Population']
    
    #to classified each state risk based on their total_cases / Population
    VeryHighRisk = outlierDetection[outlierDetection['total_cases / Population'] > (Q3['total_cases / Population'] + 1.5*IQR['total_cases / Population'])]
    print('Very High Risk States')
    if len(VeryHighRisk) == 0 :
        print('')
        print('None')
        print('')

    HighRisk = outlierDetection[outlierDetection['total_cases / Population'] >= Q3['total_cases / Population']]
    HighRisk = HighRisk[HighRisk['total_cases / Population'] < (Q3['total_cases / Population'] + IQR['total_cases / Population'])]
    print('High Risk States')

    MediumRisk = outlierDetection[outlierDetection['total_cases / Population'] >= Q1['total_cases / Population']]
    MediumRisk = MediumRisk[MediumRisk['total_cases / Population'] < Q3['total_cases / Population']]
    print('Medium Risk States')

    LowRisk = outlierDetection[outlierDetection['total_cases / Population'] < Q1['total_cases / Population']]
    print('Low Risk States')
    

    