#!/usr/bin/python

import os
import tkinter

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy import stats
from tkinter import *
from tkinter import messagebox


def train_model():
    df = pd.read_csv('heart.csv')
    z = np.abs(stats.zscore(df))
    df_clean = df[(z < 3.5).all(axis=1)]
    data = pd.get_dummies(df_clean, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])

    y = data['target']
    y = np.array(y)
    x = data.drop(columns=['target'])

    clf = ExtraTreesClassifier(n_estimators=500).fit(x, y)
    selector = SelectFromModel(clf, prefit=True)
    x_reduced = selector.transform(x)

    clf_lr = LogisticRegression(C=1, class_weight=None, penalty='l2', solver='newton-cg')
    kf = KFold(n_splits=10)

    i = 10
    for train_index, test_index in kf.split(x_reduced):
        i += 1
        # Splitting the data
        X_train, X_test = x_reduced[train_index], x_reduced[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = clf_lr.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    return model


def calculate_risk():
    should_i_calculate = 0

    entry_age_value      = (int(entry_age.get()))
    entry_pressure_value = (int(entry_pressure.get()))
    entry_chol_value     = (int(entry_chol.get()))
    entry_max_value      = (int(entry_max.get()))
    entry_angin_value    = (int(entry_angin.get()))
    entry_old_value      = (int(entry_old.get()))
    entry_cp0_value      = (int(entry_cp0.get()))
    entry_slop2_value    = (int(entry_slop2.get()))
    entry_ca0_value      = (int(entry_ca0.get()))
    entry_thal2_value    = (int(entry_thal2.get()))
    entry_thal3_value    = (int(entry_thal3.get()))


    if entry_age_value > 120 or entry_age_value < 1:
        entry_age.configure(background='red')
        entry_age.insert(END, ' : Invalid age must be between 1-120')
        should_i_calculate = 1

    elif entry_age_value:
        entry_age.configure(background='lime green')

    if entry_pressure_value > 350 or entry_pressure_value < 10:
        entry_pressure.configure(background='red')
        should_i_calculate = 1

    elif entry_pressure:
        entry_pressure.configure(background='lime green')

    if entry_chol_value > 450 or entry_chol_value < 0:
        entry_chol.configure(background='red')
        should_i_calculate = 1
    elif entry_chol_value:
        entry_chol.configure(background='lime green')

    if entry_max_value > 350 or entry_max_value < 20:
        entry_max.configure(background='red')
        should_i_calculate = 1
    elif entry_max_value:
        entry_max.configure(background='lime green')

    if entry_angin_value == 1 or entry_angin_value == 0:
        entry_angin.configure(background='lime green')
    elif entry_angin_value:
        entry_angin.configure(background='red')
        should_i_calculate = 1

    if entry_old_value < -1 or entry_old_value > 20:
        entry_old.configure(background='red')
        should_i_calculate = 1
    elif entry_old_value == 0:
        entry_old.configure(background='lime green')
    elif entry_old_value:
        entry_old.configure(background='lime green')

    if entry_cp0_value < -1 or entry_cp0_value > 10:
        entry_cp0.configure(background='red')
        should_i_calculate = 1
    elif entry_cp0_value == 0:
        entry_cp0.configure(background='lime green')
    elif entry_cp0_value:
        entry_cp0.configure(background='lime green')

    if entry_slop2_value < 0 or entry_slop2_value > 10:
        entry_slop2.configure(background='red')
        should_i_calculate = 1
    elif entry_slop2_value == 0:
        entry_slop2.configure(background='lime green')
    elif entry_slop2_value:
        entry_slop2.configure(background='lime green')

    if entry_ca0_value < -1 or entry_ca0_value > 10:
        entry_ca0.configure(background='red')
        should_i_calculate = 1
    elif entry_ca0_value == 0:
        entry_ca0.configure(background='lime green')
    elif entry_ca0_value:
        entry_ca0.configure(background='lime green')

    if entry_ca0_value < -1 or entry_ca0_value > 10:
        entry_ca0.configure(background='red')
        should_i_calculate = 1
    elif entry_ca0_value == 0:
        entry_ca0.configure(background='lime green')
    elif entry_ca0_value:
        entry_ca0.configure(background='lime green')

    if entry_thal2_value < -1 or entry_thal2_value > 10:
        entry_thal2.configure(background='red')
        should_i_calculate = 1
    elif entry_thal2_value == 0:
        entry_thal2.configure(background='lime green')
    elif entry_thal2_value:
        entry_thal2.configure(background='lime green')

    if entry_thal3_value < -1 or entry_thal3_value > 10:
        entry_thal3.configure(background='red')
        should_i_calculate = 1
    elif entry_thal3_value == 0:
        entry_thal3.configure(background='lime green')
    elif entry_thal3_value:
        entry_thal3.configure(background='lime green')

    risk_list = []
    risk_list.append(entry_age_value)
    risk_list.append(entry_pressure_value)
    risk_list.append(entry_chol_value)
    risk_list.append(entry_max_value)
    risk_list.append(entry_angin_value)
    risk_list.append(entry_old_value)
    risk_list.append(entry_cp0_value)
    risk_list.append(entry_slop2_value)
    risk_list.append(entry_ca0_value)
    risk_list.append(entry_thal2_value)
    risk_list.append(entry_thal3_value)

    if should_i_calculate == 0:
        result = model.predict_proba([risk_list])
        result = result[0][1]*100
        Label(root, text=str(round(result,2))+' %', anchor='w', justify=LEFT).grid(sticky=W, row=16, column=7, columnspan=value_column)


# Will be initialized first and only once!
model = train_model()

# Will be running until closed (mainloop)
if __name__ == "__main__":
    window_geometry = "500x500"
    window_title = 'Heart Disease Prediction\'s'

    root = tkinter.Tk()
    root.geometry(window_geometry)
    root.title(window_title)

    entry_width = 40
    value_column = 7

    Label(root, text="Patient Age [1-120]: ", anchor='w', justify=LEFT).grid(sticky = W, row=0, column=0, columnspan=value_column)
    Label(root, text="Resting blood pressure: [10-350]", anchor='w', justify=LEFT).grid(sticky = W,row=1, column=0, columnspan=value_column)
    Label(root, text="Cholesteral mg/dl: [1-450] ", anchor='w', justify=LEFT).grid(sticky = W,row=2, column=0, columnspan=value_column)
    Label(root, text="Maximum heart rate achived: [20-350] ", anchor='w', justify=LEFT).grid(sticky = W,row=3, column=0, columnspan=value_column)
    Label(root, text="Exercise induced angina 1/0 yes/no: ", anchor='w', justify=LEFT).grid(sticky = W,row=4, column=0, columnspan=value_column)
    Label(root, text="Oldpeak: [0-10] ", anchor='w', justify=LEFT).grid(sticky = W,row=5, column=0, columnspan=value_column)
    Label(root, text="CP 0: [0-10] ", anchor='w', justify=LEFT).grid(sticky = W,row=6, column=0, columnspan=value_column)
    Label(root, text="Slope 2: [0-10] ", anchor='w', justify=LEFT).grid(sticky = W,row=7, column=0, columnspan=value_column)
    Label(root, text="Ca 0: [0-10] ", anchor='w', justify=LEFT).grid(sticky = W,row=8, column=0, columnspan=value_column)
    Label(root, text="Thal 2: [0-10] ", anchor='w', justify=LEFT).grid(sticky = W,row=9, column=0, columnspan=value_column)
    Label(root, text="Thal 3: [0-10] ", anchor='w', justify=LEFT).grid(sticky = W,row=10, column=0, columnspan=value_column)
    Label(root, text='').grid(row=11)

    global entry_age
    global entry_pressure
    global entry_chol
    global entry_max
    global entry_angin
    global entry_old
    global entry_cp0
    global entry_slop2
    global entry_ca0
    global entry_thal2
    global entry_thal3

    entry_age = Entry(root, width = entry_width)
    entry_age.grid(row=0, column=7, columnspan=value_column)

    entry_pressure = Entry(root, width = entry_width)
    entry_pressure.grid(row=1, column=7, columnspan=value_column)

    entry_chol = Entry(root, width = entry_width)
    entry_chol.grid(row=2, column=7, columnspan=value_column)

    entry_max = Entry(root, width = entry_width)
    entry_max.grid(row=3, column=7, columnspan=value_column)

    entry_angin = Entry(root, width = entry_width)
    entry_angin.grid(row=4, column=7, columnspan=value_column)

    entry_old = Entry(root, width = entry_width)
    entry_old.grid(row=5, column=7, columnspan=value_column)

    entry_cp0 = Entry(root, width = entry_width)
    entry_cp0.grid(row=6, column=7, columnspan=value_column)

    entry_slop2 = Entry(root, width = entry_width)
    entry_slop2.grid(row=7, column=7, columnspan=value_column)

    entry_ca0 = Entry(root, width = entry_width)
    entry_ca0.grid(row=8, column=7, columnspan=value_column)

    entry_thal2 = Entry(root, width = entry_width)
    entry_thal2.grid(row=9, column=7, columnspan=value_column)

    entry_thal3 = Entry(root, width = entry_width)
    entry_thal3.grid(row=10, column=7, columnspan=value_column)

    calculate_button = Button(root, text='Calculate Risk', width = 40, command=calculate_risk).grid(row=12, column=1, columnspan=40)

    Label(root, text='').grid(row=13)
    Label(root, text='').grid(row=14)
    Label(root, text='').grid(row=15)

    Label(root, text="Heart disease risk probability: ", anchor='w', justify=LEFT).grid(sticky = W, row=16, column=0, columnspan=value_column)

    tkinter.mainloop()


