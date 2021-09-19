import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue
from functions import get_result
import numpy as np

st.title('An Exploration of Covid-19 in Malaysia')

cases_malaysia = pd.read_csv('./data/cases_malaysia.csv')
cases_malaysia.fillna(0, inplace=True)
cases_state = pd.read_csv('./data/cases_state.csv')
cases_state.fillna(0, inplace=True)
deaths_state = pd.read_csv('./data/deaths_state.csv')

st.write(
    '''
    Over the course of this notebook, we will try to study the trends in the Covid pandemic. This will be a comprehensive
    analysis of the data, studying trends on both a national and state-by-state level.
    '''
)

st.markdown('## Exploratory Data Analysis')
st.markdown(
    '''
    The following preprocessing steps were employed to gain a deeper understanding of the data.\n
    1. **Missing Values**: We imputed missing values with 0, because we only found null values on cases recovered column and the null is because
    they probably have not been updated yet. Using 0 was a safe assumption to avoid misleading the recovery trends. \n
    2. **Discretizing the data**: We discretized cases into 3 categories: **Low**, **Medium**, and **High** for classification problems later on.\n
    '''
)

st.write(
    '''
    The following EDA steps were performed:\n
    1. **Statistical Summary**: We used the `describe()` method to get a statistical summary of the cleaned data.\n
    '''
)
st.dataframe(cases_malaysia.describe())

st.write(
    '''
    2. **Unique States**: What are the unique states in the dataset?\n
    '''
)

st.write(
    '''
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
    Beyond just plotting them, we will identify the risk levels:\n
    1. Upper outliers are considered VERY HIGH RISK states.\n
    2. States between Q3 and Q4 are considered HIGH RISK states.\n
    3. States between Q1 and Q2 are considered MEDIUM RISK states.\n
    4. Lastly, states below Q1 are considered LOW RISK states.\n
''')
num_of_state = 16
wave1_each_states = cases_state.iloc[:170*num_of_state]
wave2_each_states = cases_state.iloc[170*num_of_state:430*num_of_state]
wave3_each_states = cases_state.iloc[430*num_of_state:]

get_result(wave1_each_states, 1, st)
get_result(wave2_each_states, 2, st)
get_result(wave3_each_states, 3, st)



# This global variable 'bar_plot' will be used later on

#q1 :

#q2 : daily new cases

#q3 : what is the reasons cause that the numbers of new cases increase
#number of import case
#number of testing

#q4 :