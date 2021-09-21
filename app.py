import streamlit as st
import pandas as pd
from streamlit.proto.PlotlyChart_pb2 import PlotlyChart
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue
from functions import get_result, get_best_features, best_features_state, linear_regression, random_forest_regressor, svm_regression
import numpy as np
import plotly.express as px
from plotly.graph_objects import Heatmap
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

clusters = pd.read_csv('./data/clusters.csv')
clusters.reset_index(inplace=True)
cluster_cases = clusters.pivot_table(index = 'category', values = ['cases_total', 'tests'], aggfunc = 'sum')

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
    1. **Statistical Summary**: We used obtained a full statistical summary of the cleaned and transformed data.\n
    2. **Case Bar Charts Per Wave**: It is misleading to treat the entire pandemic as one time-series. The range of values are almost completely different, so we study trends per wave.\n
    3. **Analysis of Covid-19 clusters**: A well-known phenomenon known to cause sudden spikes.
    4. **Linking time-series spikes with real events**: When cases rise suddenly on a particular day, there's usually a reason.
    '''
)

st.markdown('''
    ### How have different states performed over the 3 waves in the pandemic?
''')

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
    We first group the data by state and wave. Then we divide the data by the population of the state
    to account for the different sizes, afterwhich the column is scaled between 0 and 1. A state with an abnormally different number of cases in a wave is considered an outlier.
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

st.markdown('''
    ## What does the general trend look like across the pandemic? Can spikes be attributed to certain events?
    We can do a complete time series graph to observe the general trend and whether there have been sharp increases.
''')

plot = px.line(cases_malaysia, x='date', y='cases_new')
st.plotly_chart(plot)
st.markdown('''
    In the news, we usually see the sudden rise in cases last year attributed to the Sabah election which took place on September 26 2020.
    We asked ourselves whether there was an observable increase around that time. The graph shows that around early October, the scale of the cases
    start to change incrementally, and almost form an exponential curve since.
    Daily new cases have 3 other relative peaks:\n
    1. January 30 2021: 5725 cases\n
    2. May 29 2021: 9020 cases\n
    3. August 26 2021: 24599 cases\n
''')

st.markdown('''
    ## How influential are different clusters in the country?
    **"Kluster Langkawi", "Kluster Mamak", "Kluster Mahkamah"**\n
    We often hear these terms in the news. A cluster refers to an aggregation of cases of a disease given the
    contagious nature of Covid-19. When there are sudden spikes, these can often be attributed to a new cluster, but what are these clusters
    and which clusters are more influential than others?
''')

st.table(cluster_cases)

col1, col2 = st.columns(2)

with col1:
    col1.subheader('Cases Total')
    cluster_cases_box = px.box(cluster_cases, x='cases_total', width=800, height=300)
    st.plotly_chart(cluster_cases_box)

with col2:
    col2.subheader('Tests')
    cluster_cases_tests = px.box(cluster_cases, x='tests', width=500, height=300)
    st.plotly_chart(cluster_cases_tests)

grouped_clusters = go.Figure(data=[
    go.Bar(name='Total Cases', x=cluster_cases.index, y=cluster_cases['cases_total']),
    go.Bar(name='Tests', x=cluster_cases.index, y=cluster_cases['tests'])
])
grouped_clusters.update_layout(barmode='group')
st.plotly_chart(grouped_clusters)

st.markdown('''
    The first observation we can make is that with more testing, there are more cases, given how similar
    the shape of the 2 boxplots are. There is a right skew, indicating that the majority of clusters are on the upper
    end of the scale. Workplace clusters prove to be the most dramatic, sitting as upper outliers in both graphs. We can perhaps
    attribute this to inaction and many businesses staying open during the different MCOs (e.g. Top Glove).

    Religious clusters, that often appear in the news, are the lowest of all. Maybe religious clusters are easier to regulate and take
    swift action, which may just be a matter of shutting down the institution.
''')

st.markdown(
'''
    ## How do the attributes of different states correlate with Pahang and Johor?\n
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
The top 6 features for Pahang are: tests, pkrc_admitted_covid, icu_covid, vent_covid and hosp_covid.
''')

st.markdown('Kedah')
get_best_features('Kedah', states_allfeatures['Kedah'], st, display_scatter_plots=1)
st.write('''
The top 6 features for Kedah are: pkrc_covid, icu_covid, beds_covid, admitted_covid and hosp_covid.
''')

st.markdown('Johor')
get_best_features('Johor', states_allfeatures['Johor'], st, display_scatter_plots=1)
st.write('''
The top 6 features for Johor are: pkrc_covid, icu_covid, beds_covid, admitted_covid and hosp_covid.
''')

st.markdown('Selangor')
get_best_features('Selangor', states_allfeatures['Selangor'], st, display_scatter_plots=1)
st.write('''
The top 6 features for Selangor are: vent_covid, pkrc_covid, icu_covid, beds_covid, admitted_covid and hosp_covid.
''')

st.markdown('''
    ## Modelling for Different States
    ### Classification and Regression Models
    Models will not be run on Streamlit to make sure this page loads fast. For
    alll training and hyperparameters, refer to the notebook.

    For regression models, all relevant attributes are initially scaled.
    For classification, we calculate the weighted averaged F1 score that takes into account the class imbalance.
'''
)

state_select = st.selectbox('Select a state.', ['Pahang', 'Kedah', 'Johor', 'Selangor'], key='state')

if state_select == 'Pahang':
    pahang_regression = {
        'model': ['Support Vector Regression', 'Linear Regression', 'Random Forest Regression'],
        'accuracy': [0.84037, 0.90272, 0.91098],
        'Mean Squared Error': [0.00820, 0.00510, 0.00457]
    }
    pahang_classification = pd.DataFrame(pahang_regression)
    st.table(pahang_regression)

    pahang_classification = {
        'model': ['Support Vector Classification', 'Random Forest Classification', 'Logistic Regression'],
        'accuracy': [0.9286, 0.9143, 0.9429],
        'Weighted averaged F1 Score': [0.9134, 0.91448, 0.9206]
    }
    st.table(pahang_classification)

    st.markdown('''
        For Pahang, classification models generally do better because there is a lower chance of making a mistake.
        They can be only 1 of 3 classes (Low, Medium and High).
        The best performing Classification model is surprisingly Logistic Regression and the best performing regression
        model is Random Forest Regression.
    '''
    )

elif state_select == 'Kedah':
    kedah_regression = {
        'model': ['Support Vector Regression', 'Linear Regression', 'Random Forest Regression'],
        'accuracy': [0.95649, 0.94769, 0.9663],
        'Mean Squared Error': [0.0029, 0.0035, 0.0023]
    }
    st.table(kedah_regression)

    kedah_classification = {
        'model': ['Support Vector Classification', 'Random Forest Classification', 'Logistic Regression'],
        'accuracy': [0.9265, 0.9412, 0.8971],
        'Weighted averaged F1 Score': [0.9253, 0.9411, 0.8653]
    }
    st.table(kedah_classification)

    st.markdown('''
        The best performing Classification model is Random Forest Classification and the best performing regression
        model is Random Forest Regression. The reason for this is maybe that Random Forest Regression is more robust to
        outliers.
    '''
    )

elif state_select == 'Johor':
    johor_regression = {
        'model': ['Support Vector Regression', 'Linear Regression', 'Random Forest Regression'],
        'accuracy': [0.84401, 0.86624, 0.90511],
        'Mean Squared Error': [0.00618, 0.00530, 0.00376]
    }
    st.table(johor_regression)

    johor_classification = {
        'model': ['Support Vector Classification', 'Random Forest Classification', 'Logistic Regression'],
        'accuracy': [0.95385, 0.93846, 0.93846],
        'Macro averaged F1 Score': [0.9527, 0.9385, 0.9234]
    }
    st.table(johor_classification)

    st.markdown('''
        Once again, Random Forest Regression prevails in the regression arena with the highest accuracy and lowest MSE.
        Support Vector Classification is the best performing classification model in this case.
    ''')

elif state_select == 'Selangor':
    selangor_regression = {
        'model': ['Support Vector Regression', 'Linear Regression', 'Random Forest Regression'],
        'accuracy': [0.94619, 0.96082, 0.9566],
        'Mean Squared Error': [0.0029, 0.0035, 0.0023]
    }
    st.table(selangor_regression)

    selangor_classification = {
        'model': ['Support Vector Classification', 'Random Forest Classification', 'Logistic Regression'],
        'accuracy': [0.97170, 0.93396, 0.96226],
        'Macro averaged F1 Score': [0.97229, 0.93240, 0.95952]
    }
    st.table(selangor_classification)
    st.markdown('''
        The best performing Regression model for Selangor is Linear Regression, which may suggest a stronger linear relationship between the
        features and the target. The best classification model is Support Vector Classification. One intersting observation is that Selangor's cases
        are mostly High, so F1 score is a safer metric to use than accuracy to decide.
    ''')