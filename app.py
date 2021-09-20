import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue
from functions import get_result, get_best_features, best_features_state, linear_regression, random_forest_regressor, svm_regression
import numpy as np
import plotly.express as px
from plotly.graph_objects import Heatmap
from plotly.subplots import make_subplots

st.title('An Exploration of Covid-19 in Malaysia')

cases_malaysia = pd.read_csv('./data/cases_malaysia.csv')
cases_malaysia.fillna(0, inplace=True)
cases_state = pd.read_csv('./data/cases_state.csv')
cases_state.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1'], inplace=True)

cases_state_pivoted = cases_state.pivot_table(index='date', columns='state', values='cases_new')
cases_state.fillna(0, inplace=True)
deaths_state = pd.read_csv('./data/deaths_state.csv')
deaths_state.fillna(0, inplace=True)

deaths_pivoted = deaths_state.pivot_table(index='date', columns='state', values='deaths_new')
tests_state = pd.read_csv('./data/tests_state.csv')
tests_pivoted = tests_state.pivot(index='date', columns='state', values='total')
tests_state.fillna(0, inplace=True)
tests_state.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1'], inplace=True)
tests_state.rename(columns={'total': 'tests'}, inplace=True)

cases_state = pd.read_csv('./data/cases_state.csv', index_col=0)
cases_state_pivoted = cases_state.pivot(index='date', columns='state', values='cases_new')

deaths_state = pd.read_csv('./data/deaths_state.csv')
deaths_state_pivoted = deaths_state.pivot(index='date', columns='state', values='deaths_new')

tests_state = pd.read_csv('./data/tests_state.csv')
tests_state_pivoted = tests_state.pivot(index='date', columns='state', values='total')
tests_state.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1', 'Unnamed: 0.1.1.1.1'], inplace=True)
tests_state.rename(columns={'total': 'tests'}, inplace=True)

quarantine_state = pd.read_csv('./data/pkrc.csv')
quarantine_state.fillna(0, inplace=True)
quarantine_state_pivoted = quarantine_state.pivot(index='date', columns='state', values='admitted_covid')
discharge_quarantine_state_pivoted = quarantine_state.pivot(index='date', columns='state', values='discharge_covid')
quarantine_state.rename(columns={'beds': 'pkrc_beds', 'admitted_pui': 'pkrc_admitted_pui', 'admitted_covid': 'pkrc_admitted_covid', 'discharge_covid': 'pkrc_discharge_covid', 'admitted_total': 'pkrc_admitted_total'}, inplace=True)

icu_state = pd.read_csv('./data/icu.csv')
icu_state_pivoted = icu_state.pivot(index='date', columns='state', values='icu_covid')
hospital = pd.read_csv('./data/hospital.csv')
hospital_admitted_pivoted = hospital.pivot(index='date', columns='state', values='admitted_covid')
hospital_discharged_pivoted = hospital.pivot(index='date', columns='state', values='discharged_covid')
population_state = pd.read_csv('./data/population.csv', index_col=0)

states_list = ['Pahang', 'Kedah', 'Johor', 'Selangor']
states = {}
states_allfeatures = {}

for state in states_list:
    df = pd.DataFrame()
    df['cases'] = cases_state_pivoted[state]
    df['deaths'] = deaths_state_pivoted[state]
    df['tests'] = tests_state_pivoted[state]
    df['quarantine'] = quarantine_state_pivoted[state]
    df['discharge_quarantine'] = discharge_quarantine_state_pivoted[state]
    df['icu'] = icu_state_pivoted[state]
    df['hospital_admitted'] = hospital_admitted_pivoted[state]
    df['hospital_discharged'] = hospital_discharged_pivoted[state]
    df.fillna(0, inplace=True)
    states[state] = df

for state in states_list:
    df = cases_state[cases_state['state'] == state]
    df = df.merge(deaths_state[deaths_state['state'] == state], how='left')
    df.fillna(0, inplace=True)
    df = df.merge(tests_state[tests_state['state'] == state], how='left')
    df.fillna(0, inplace=True)
    df = df.merge(quarantine_state[quarantine_state['state'] == state], how='left')
    df.fillna(0, inplace=True)
    df = df.merge(icu_state[icu_state['state'] == state], how='left')
    df.fillna(0, inplace=True)
    df = df.merge(hospital[hospital['state'] == state], how='left')
    df.fillna(0, inplace=True)
    df = df.merge(tests_state[tests_state['state'] == state], how='left')
    df.fillna(0, inplace=True)
    df.set_index('date', inplace=True)
    states_allfeatures[state] = df

st.write(
    '''
    Over the course of this notebook, we will try to study the trends in the Covid pandemic. This will be a comprehensive
    analysis of the data, studying trends on both a national and state-by-state level.
    '''
)

st.markdown('## Preprocessing')
st.markdown(
    '''
    The following preprocessing steps were employed to gain a deeper understanding of the data.\n
    1. **Missing Values**: We imputed missing values with 0, because we only found null values on cases recovered column and the null is because
    they probably have not been updated yet. Using 0 was a safe assumption to avoid misleading the recovery trends. \n
    2. **Discretizing the data**: We discretized cases into 3 categories: **Low**, **Medium**, and **High** for classification problems later on.\n
    '''
)

st.markdown('## Exploratory Data Analysis')
st.write(
    '''
    The following EDA steps were performed:\n
    1. **Statistical Summary**: We used the `describe()` method to get a statistical summary of the cleaned data.\n
    2. **Case Bar Charts Per Wave**: It is misleading to treat the entire pandemic as one time-series. The range of values are almost completely different, so we study trends per wave.\n
    '''
)

st.markdown('''
### The First Wave: Covid-19 in 2020
#### Haunting Malaysia from 2020-01-25 to 2020-07-12
''')
st.markdown('\n\n')

wave1 = cases_malaysia.iloc[:170]
# wave1.set_index('date', inplace=True)

wave1_bar = alt.Chart(wave1).mark_bar().encode(
    x=alt.X('date', axis=alt.Axis(title='Date', labels=False)),
    y=alt.Y('cases_new', axis=alt.Axis(title='Cases')),
).properties(
    width=600,
)
st.altair_chart(wave1_bar)

st.markdown('''
### The Second Wave
#### Haunting Malaysia from 2020-07-13 to 2021-3-29
''')
st.markdown('\n\n')

wave2 = cases_malaysia.iloc[170:340]
wave2_bar = alt.Chart(wave2).mark_bar().encode(
    x=alt.X('date', axis=alt.Axis(title='Date', labels=False)),
    y=alt.Y('cases_new', axis=alt.Axis(title='Cases')),
).properties(
    width=600,
)
st.altair_chart(wave2_bar)

st.markdown('''
### The Third Wave: Covid-19
#### Haunting Malaysia from 2020-03-30 to Now
''')
st.markdown('\n\n')
wave3 = cases_malaysia.iloc[340:]
wave3_bar = alt.Chart(wave3).mark_bar().encode(
    x=alt.X('date', axis=alt.Axis(title='Date', labels=False)),
    y=alt.Y('cases_new', axis=alt.Axis(title='Cases')),
).properties(
    width=600,
)
st.altair_chart(wave3_bar)

st.markdown('''
Each of the waves are on a completely different scale as we can observe. Wave 1 hits a maximum of **300** cases a day,
the second wave hits around **6000** and the third waves reaches a resounding **25000** cases. This is representative of an exponential
increase.
''')

st.markdown('''
    ### Outliers
    We first group the data by state and wave. Then we normalise the data by the population of the state
    to account for the different sizes. A state with an abnormally different number of cases in a wave is considered an outlier.
''')
num_of_state = 16
wave1_each_states = cases_state.iloc[:170*num_of_state]
wave2_each_states = cases_state.iloc[170*num_of_state:430*num_of_state]
wave3_each_states = cases_state.iloc[430*num_of_state:]

get_result(wave1_each_states, 1, st)
st.markdown('''
    It's quite clear that in the first wave, the majority of the country is in the
    lower end. However, there are a few states that stand out as outliers, and these come as no surprise: Negeri Sembilan,
    W.P. Kuala Lumpur and Putrajaya.
    Note that this also accounts for their population, which is why Putrajaya ranks higher than Selangor itself.
    Kedah, Kelantan, Perlis and Sabah are relatively low risk.
    '''
)

get_result(wave2_each_states, 2, st)
st.markdown('''
    By the second wave, most of the country has caught up and there are no more outliers. There is a right skew,
    so the majority of states are now in the upper end of the range.
''')

get_result(wave3_each_states, 3, st)
st.markdown('''
    In the third wave, the distribution is almost symmetric. The case number for the worst hit state has increased, as observed earlier.
    More of the country is at a high risk level, with the exception of perhaps Perlis, which stayed consistently low across all 3 waves.
''')

st.markdown(
'''
    ## What states exhibit strong correlation with (i) Pahang and (ii) Johor?\n
    We can study correlation based on case number, deaths and testing. It doesn't matter if
    they are on different scales as long as they change in a similar pattern.
''')

cases_correlation = cases_state_pivoted.corr()
tests_correlation = tests_pivoted.corr()
deaths_correlation = deaths_pivoted.corr()

st.markdown('### Correlation by Cases')
cases_heatmap = px.imshow(cases_correlation)
st.plotly_chart(cases_heatmap)
st.write('Kedah, Terrengganu and Perak exhibit strong correlation with Pahang in daily new cases.')
cases_correlation_pahang = cases_correlation['Pahang'].sort_values(ascending=False)
cases_correlation_pahang = pd.DataFrame(cases_correlation_pahang)
st.write(cases_correlation_pahang.iloc[1:4])
st.write('''
    Pulau Pinang, Terengganu and Perak are highly correlated with Johor in terms of new daily cases.
''')
cases_correlation_Johor = cases_correlation['Johor'].sort_values(ascending=False)
cases_correlation_Johor = pd.DataFrame(cases_correlation_Johor)
st.write(cases_correlation_Johor.iloc[1:4])

st.markdown('### Correlation by Tests')
tests_heatmap = px.imshow(tests_correlation)
st.plotly_chart(tests_heatmap)
st.write('Kedah, Johor and Perak exhibit strong correlation with Pahang in terms of testing per day.')
tests_correlation_pahang = tests_correlation['Pahang'].sort_values(ascending=False)
tests_correlation_pahang = pd.DataFrame(tests_correlation_pahang)
st.write(tests_correlation_pahang.iloc[1:4])

st.write('''
    Kedah, Perlis and Pulau Pinang are strongly correlated with Johor.
''')
tests_correlation_Johor = tests_correlation['Johor'].sort_values(ascending=False)
tests_correlation_Johor = pd.DataFrame(tests_correlation_Johor)
st.write(tests_correlation_Johor.iloc[1:4])

st.markdown('### Correlation by Deaths')
deaths_heatmap = px.imshow(deaths_correlation)
st.plotly_chart(deaths_heatmap)
st.write('Negeri Sembilan, Johor and Selangor exhibit a strong correlated with Pahang in terms of deaths.')
deaths_correlation_pahang = deaths_correlation['Pahang'].sort_values(ascending=False)
deaths_correlation_pahang = pd.DataFrame(deaths_correlation_pahang)
st.write(deaths_correlation_pahang.iloc[1:4])

st.write('Pulau Pinang, Sabah and Perak exhibit strong correlation with Pahang in daily new deaths.')
deaths_correlation_Johor = deaths_correlation['Johor'].sort_values(ascending=False)
deaths_correlation_Johor = pd.DataFrame(deaths_correlation_Johor)
st.write(deaths_correlation_Johor.iloc[1:4])


st.markdown(
'''
    ## Strong features/indicators to daily cases
    For feature selection, we first handpick features we think is relevant from the full feature set
    and then we delegate to 3 feature selection algorithms to further narrow done features for each state.
    Several algorithms are used and a vote is then taken to select the final feature set.
    The 3 algorithms are:\n
    1. BORUTA\n
    2. Mutual Info Regression\n
    4. Recursive Feature Elimination (RFE)
''')

st.markdown('Pahang')
get_best_features('Pahang', states_allfeatures['Pahang'], st, display_scatter_plots=1)
st.markdown('''
We know that the four best features are quarantine, discharge_quarantine, icu and hospital_admitted.
For quarantine and discharge_quarantine, we can see that there is a strong, positive and linear relationship.
''')

st.markdown('Kedah')
get_best_features('Kedah', states_allfeatures['Kedah'], st, display_scatter_plots=1)
st.write('''
The 4 best features are hospital_discharged, icu, discharge_quarantine and hospital_admitted.
icu and hospital_admitted have a stronger fit, but the other features have passed the feature selection tests.
''')

st.markdown('Johor')
get_best_features('Johor', states_allfeatures['Johor'], st, display_scatter_plots=1)
st.write('''
Write description
''')

st.markdown('Selangor')
get_best_features('Selangor', states_allfeatures['Selangor'], st, display_scatter_plots=1)
st.write('''
Write description
''')

st.markdown('''
    ## Modelling
    ### Regression Models
'''
)
regression_functions = {
    'SVR': svm_regression,
    'Linear Regression': linear_regression,
    'Random Forest Regressor': random_forest_regressor
}
regressor_select = st.selectbox('Select a regressor to run.', ['SVR', 'Linear Regression', 'Random Forest Regressor'])

pahang_regressor = regression_functions[regressor_select](states_allfeatures['Pahang'], best_features_state['Pahang'])
kedah_regressor = regression_functions[regressor_select](states_allfeatures['Kedah'], best_features_state['Kedah'])
johor_regressor = regression_functions[regressor_select](states_allfeatures['Johor'], best_features_state['Johor'])
selangor_regressor = regression_functions[regressor_select](states_allfeatures['Selangor'], best_features_state['Selangor'])

st.write(pahang_regressor)

st.markdown('''
## Modelling
### Classification Models
''')
classifier_select = st.selectbox('Select a classifier to run.', ['SVM', 'Random Forest', 'Logistic Regression', 'Decision Tree'])
classification_functions = {}
