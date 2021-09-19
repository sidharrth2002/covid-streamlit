import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_regression, RFE, VarianceThreshold
from sklearn.linear_model import LinearRegression
from plotly.subplots import make_subplots
import sklearn

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

    print(outlierDetection)
    fig = px.box(outlierDetection.rename(columns={'total_cases / Population': 'Adjusted Case Number'}), x="Adjusted Case Number", title='Wave '+ str(wave_num))
    st.plotly_chart(fig)

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

#the function we use to find the best features
def get_best_features(df, st, display_scatter_plots = 0):
    X = df.drop(['cases'], axis=1)
    y = df['cases']

    selector = VarianceThreshold(3)
    selector.fit(df)
    variance_best = df.columns[selector.get_support()]

    selector = SelectKBest(mutual_info_regression, k=4)
    selector.fit(X, y)
    mutual_info_best = X.columns[selector.get_support()]

    selector = SelectKBest(chi2, k=4)
    selector.fit(X, y)
    chi2_best = X.columns[selector.get_support()]

    rfe_selector = RFE(LinearRegression(), n_features_to_select=4)
    rfe_selector.fit(X, y)
    rfe_best = X.columns[rfe_selector.get_support()]

    columns = df.columns
    columns_count = {}
    for column in columns:
        columns_count[column] = list(variance_best).count(column) + list(mutual_info_best).count(column) + list(chi2_best).count(column) + list(rfe_best).count(column)

    best_features = sorted(columns_count, key=columns_count.get)[-4:]
    grid_pos = [[1, 1], [1, 2], [2, 1], [2, 2]]

    if display_scatter_plots == 1:
        print("The best four features")
        print("")
        feature_subplots = make_subplots(rows=2, cols=2, subplot_titles=best_features)
        for num in range(len(best_features)):
            sns.scatterplot(x= best_features[num], y="cases", data=df)
            z = np.polyfit(df[best_features[num]], df['cases'], 1)
            p = np.poly1d(z)
            feature_subplots.add_trace(go.Scatter(x=df[best_features[num]], y=df['cases'], mode='markers', name=best_features[num], line=go.scatter.Line()), row=grid_pos[num][0], col=grid_pos[num][1])

    if 'tests' not in best_features:
        best_features.append('tests')

    st.plotly_chart(feature_subplots)

    return best_features