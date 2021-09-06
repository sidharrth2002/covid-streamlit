import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('An Exploration of Covid-19 in Malaysia')

cases_malaysia = pd.read_csv('./data/cases_malaysia.csv')
cases_state = pd.read_csv('./data/cases_state.csv')
deaths_state = pd.read_csv('./data/deaths_state.csv')



cases_state = cases_state.pivot(index='date', columns='state', values='cases_new')
st.write('Correlations')
cases_correlations = cases_state.corr(method='pearson')

fig, ax = plt.subplots(figsize=(20, 20))
st.markdown('### Correlations between Cases Among States')
sns.heatmap(cases_correlations, annot=True, ax=ax)
st.write(fig)