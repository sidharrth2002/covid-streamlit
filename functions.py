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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, mean_squared_error

best_features_state = {
    'Pahang': ['tests', 'pkrc_admitted_covid', 'pkrc_covid', 'icu_covid', 'vent_covid', 'hosp_covid'],
    'Kedah': ['vent_covid', 'pkrc_covid', 'icu_covid', 'beds_covid', 'admitted_covid', 'hosp_covid'],
    'Selangor': ['vent_covid', 'pkrc_beds', 'icu_covid', 'beds_covid', 'admitted_covid', 'hosp_covid'],
    'Johor': ['pkrc_beds', 'pkrc_covid', 'icu_covid', 'vent_covid', 'beds_covid', 'hosp_covid']
}

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

def get_best_features(state_name, df, st, display_scatter_plots = 0):
    best_features_state = {
        'Pahang': ['tests', 'pkrc_admitted_covid', 'pkrc_covid', 'icu_covid', 'vent_covid', 'hosp_covid'],
        'Kedah': ['vent_covid', 'pkrc_covid', 'icu_covid', 'beds_covid', 'admitted_covid', 'hosp_covid'],
        'Selangor': ['vent_covid', 'pkrc_beds', 'icu_covid', 'beds_covid', 'admitted_covid', 'hosp_covid'],
        'Johor': ['pkrc_beds', 'pkrc_covid', 'icu_covid', 'vent_covid', 'beds_covid', 'hosp_covid']
    }
    best_features = best_features_state[state_name]

    grid_pos = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3]]

    if display_scatter_plots == 1:
        print("The best ten features")
        print("")
        feature_subplots = make_subplots(rows=2, cols=3, subplot_titles=best_features)
        for num in range(len(best_features)):
            sns.scatterplot(x= best_features[num], y="cases_new", data=df)
            z = np.polyfit(df[best_features[num]], df['cases_new'], 1)
            p = np.poly1d(z)
            feature_subplots.add_trace(go.Scatter(x=df[best_features[num]], y=df['cases_new'], mode='markers', name=best_features[num], line=go.scatter.Line()), row=grid_pos[num][0], col=grid_pos[num][1])

    if 'tests' not in best_features:
        best_features.append('tests')

    st.plotly_chart(feature_subplots)

    return best_features


def svm_regression(df, features):
    X = df[features]
    X = MinMaxScaler().fit_transform(X)
    y = df['cases_new']
    y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    svr_rbf = SVR(kernel='rbf', C=1e5, gamma=0.001)
    svr_rbf.fit(X_train, y_train)
    svr_rbf_pred = svr_rbf.predict(X_test)
    print(y_test.ravel().shape)

    svr_rbf_score = svr_rbf.score(X_test, y_test)
    svr_rbf_mse = mean_squared_error(y_test, svr_rbf_pred.ravel())

    actual_vs_pred = pd.DataFrame({'actual': y_test.ravel(), 'pred': pd.Series(svr_rbf_pred)})

    print('SVR RBF Score: ', svr_rbf_score)
    print('SVR RBF MSE: ', svr_rbf_mse)
    print('')

    return {
        'accuracy': svr_rbf_score,
        'mse': svr_rbf_mse,
        'results': actual_vs_pred,
    }


def linear_regression(df, features):
    X = df[features]
    X = MinMaxScaler().fit_transform(X)
    y = df['cases_new']
    y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    lin_reg_pred = lin_reg.predict(X_test)

    print(lin_reg_pred.ravel().shape)

    lin_reg_score = lin_reg.score(X_test, y_test)
    lin_reg_mse = mean_squared_error(y_test, lin_reg_pred)

    actual_vs_pred = pd.DataFrame({'actual': y_test.ravel(), 'pred': lin_reg_pred.ravel()})

    print('Linear Regression Score: ', lin_reg_score)
    print('Linear Regression MSE: ', lin_reg_mse)
    print('')

    return {
        'accuracy': lin_reg_score,
        'mse': lin_reg_mse,
        'results': actual_vs_pred,
    }

def random_forest_regressor(df, features):
    X = df[features]
    X = MinMaxScaler().fit_transform(X)
    y = df['cases_new']
    y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    rf_reg = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_reg.fit(X_train, y_train)
    rf_reg_pred = rf_reg.predict(X_test)

    rf_reg_score = rf_reg.score(X_test, y_test)
    rf_reg_mse = mean_squared_error(y_test, rf_reg_pred)

    actual_vs_pred = pd.DataFrame({'actual': y_test.ravel(), 'pred': rf_reg_pred})

    print('Random Forest Regressor Score: ', rf_reg_score)
    print('Random Forest Regressor MSE: ', rf_reg_mse)
    print('')

    return {
        'accuracy': rf_reg_score,
        'mse': rf_reg_mse,
        'results': actual_vs_pred,
    }

def print_result_for_regression_models(state) :
    svm_regression(states[state], get_best_features(states[state]))
    linear_regression(states[state], get_best_features(states[state]))
    random_forest_regressor(states[state], get_best_features(states[state]))

def supportvectormachine_classification(df, features):
    X = df[features]
    X = MinMaxScaler().fit_transform(X)
    y = df['cases_binned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    svc = SVC(kernel='rbf', C=1e5, gamma=0.001)
    svc.fit(X_train, y_train)
    svc_pred = svc.predict(X_test)

    svc_score = svc.score(X_test, y_test)
    svc_f1_score = f1_score(y_test, svc_pred, average='weighted')

    actual_vs_pred = pd.DataFrame({'actual': y_test.ravel(), 'pred': svc_pred})

    # print classification report
    class_report = classification_report(y_test, svc_pred)
    print(class_report)

    print('SVC Score: ', svc_score)
    print('SVC F1 Score: ', svc_f1_score)
    print('')

    return {
        'accuracy': svc_score,
        'f1_score': svc_f1_score,
        'results': actual_vs_pred,
        'classification_report': class_report
    }

def randomforest_classification(df, features):
    X = df[features]
    X = MinMaxScaler().fit_transform(X)
    y = df['cases_binned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    rf_clf = RandomForestClassifier(n_estimators=5, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_clf_pred = rf_clf.predict(X_test)

    rf_clf_score = rf_clf.score(X_test, y_test)
    rf_clf_f1_score = f1_score(y_test, rf_clf_pred, average='weighted')

    actual_vs_pred = pd.DataFrame({'actual': y_test.ravel(), 'pred': rf_clf_pred})

    # print classification report
    class_report = classification_report(y_test, rf_clf_pred)
    print(class_report)

    print('Random Forest Classification Score: ', rf_clf_score)
    print('Random Forest Classification F1 Score: ', rf_clf_f1_score)
    print('')

    return {
        'accuracy': rf_clf_score,
        'f1_score': rf_clf_f1_score,
        'results': actual_vs_pred,
        'classification_report': class_report
    }

def logistic_regression(df, features):
    X = df[features]
    X = MinMaxScaler().fit_transform(X)
    y = df['cases_binned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)
    log_reg_pred = log_reg.predict(X_test)

    log_reg_score = log_reg.score(X_test, y_test)
    log_reg_f1_score = f1_score(y_test, log_reg_pred, average='weighted')

    actual_vs_pred = pd.DataFrame({'actual': y_test.ravel(), 'pred': log_reg_pred})

    # print classification report
    class_report = classification_report(y_test, log_reg_pred)
    print(classification_report(y_test, log_reg_pred))

    print('Logistic Regression Score: ', log_reg_score)
    print('Logistic Regression F1 Score: ', log_reg_f1_score)
    print('')

    return {
        'accuracy': log_reg_score,
        'f1_score': log_reg_f1_score,
        'results': actual_vs_pred,
        'classification_report': class_report
    }