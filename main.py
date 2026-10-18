# Sabrina Chou
# TAC 216
# Final Project
# Description: This program uses data from Spain's Instituto Nacional de Estadística (INE) to analyze gender wage gaps across different regions in Spain.
# Users select a region to view a summary table and visualization comparing male and female incomes from 2008 to 2022.
# The program also splits the data into training and testing sets to build a neural network model to predict gender based on income. Users can choose to runt he model on their selected region or the entire dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler

def main():

    # data wrangling
    data = pd.read_csv("spain_salary_distrib.csv") # load dataset
    data = data.drop_duplicates() # remove duplicate rows
    data = data.dropna() # remove rows with missing values
    data_mean = data[data["statistic"] == "mean"] # keep only mean income rows for wage visualization
    data_mean = data_mean[["year", "state", "gender", "gross_income_euros"]] # keep only needed columns
    data = data[["year", "state", "gender", "gross_income_euros"]] # keep only needed columns

    # user picks a region
    # allow the user to choose what subsets of the data (in this case region) they would like to see (via command line prompts), and the program will serve them a visualization of that data
    regions = sorted(data["state"].unique()) # get all unique region names and sort them alphabetically
    print("This data comes from Spain's Instituto Nacional de Estadística (INE) from 2008 to 2022.")
    print("You will be able to visualize wage gaps within a region in Spain....")
    print("Pick a region:")
    print()
    for i in range(len(regions)): # loop through each region
        print(str(i + 1) + ". " + regions[i]) # print number and region name
    print()
    choice = input("Enter the number of the region you want to analyze: ") # get user's region choice
    # wnat to prevent my code from crashing if they enter a number that is not one of the options
    while not choice.isdigit(): # checking first to make sure it's an int and then turning choice into an int
        choice = input("Enter the number of the region you want to analyze: ") # get user's region choice
    choice = int(choice)
    while choice < 1 or choice > len(regions):
        print("Invalid choice, try again please.")
        choice = int(input("Enter the number of the region you want to analyze: "))
    selected_region = regions[choice - 1] # convert number to actual region name (subtract 1 since lists start at 0)
    print("You selected: " + selected_region)
    print()

    # filter the big dataset down to just chosen region, split into male and feamle table so you can get the gap
    # scientific computation: wage gap summary table
    region_data = data_mean[data_mean["state"] == selected_region] # filter for selected region
    region_data = region_data[region_data["gender"].isin(["male", "female"])] # male and female only
    region_data = region_data.sort_values("year") # sort chronologically

    male_data = region_data[region_data["gender"] == "male"].reset_index(drop=True) # male rows, but reset the row numbers/indices so that its 0 1 2 3....
    female_data = region_data[region_data["gender"] == "female"].reset_index(drop=True) # female rows, but reset the row numbers/indices so that its 0 1 2 3....

    gap = male_data["gross_income_euros"].values - female_data["gross_income_euros"].values # wage gap per year
    years = female_data["year"].values # year values for plotting could be either male or female just want the years

    # create a summary table
    summary = pd.DataFrame({"Year" : years, "Male Gross Income:" : male_data["gross_income_euros"], "Female Gross Income:" : female_data["gross_income_euros"], "Wage Gap": gap})
    print(summary) # print the table
    
    # figure 1: wage gap visualization
    # create the figure and a single axis object
    fig, ax = plt.subplots(figsize=(10, 6)) # create window (fig) and the empty chart (ax)

    # plot the male and female data on the ax
    ax.plot(years, male_data["gross_income_euros"], marker="d", color="lightseagreen", label="Male Mean Income") # draw line connecting dots for men's income, year on x-axis and gross_income_euros on y-axis
    ax.plot(years, female_data["gross_income_euros"], marker="d", color="pink", label="Female Mean Income")

    # fill the area between the two lines to highlight the gap
    ax.fill_between(years, male_data["gross_income_euros"], female_data["gross_income_euros"], color="tan", alpha=0.2) # shades the space between the two lines

    # labeling everything
    ax.set_title(f"Wage Gap Trends in: " + selected_region) # chart title with the region name
    ax.set_xlabel("Year") # x-axis label
    ax.set_ylabel("Gross Income (Euros)") # y-axis label
    ax.legend() # show the legend so you now which line is which
    ax.grid(True, linestyle='--') # make the background grid look cuter, show the grid lines, make the lines aesthetic :)

    # save and show
    fig.savefig("wage_gap_trends.png") # save the fig

    # ask user again for input of if they want the ML model to use the entire dataset or their previous selected region
    second_choice = input("\nDo you want to use the entire dataset (respond A) or your selected region (respond B) to see if income can distinguish between male and female wages? ")
    # use if else to use entire dataset or just chosen region
    while second_choice not in ["A", "B"]: # keep looping until they type A or B
        if second_choice not in ["A", "B"]: # if they typed something else
            print("Invalid choice. Please respond with A or B.")
            second_choice = input("\nDo you want to use the entire dataset (respond A) or your selected region (respond B) to see if income can distinguish between male and female wages? ")
    
    if second_choice == "A":
        data = data[data["gender"].isin(["male", "female"])] # add this filter because there is a "both" category in the dataset messing up and giving us 3 classes instead of 2
        X = data[["gross_income_euros"]] # predicters: income values
        y = data["gender"]# prediction: male or female
        print("\nNow we are going to see if income can be used to predict gender using a machine learning model for all regions...")
    elif second_choice == "B":
        X = region_data[["gross_income_euros"]] # predicters: income values
        y = region_data["gender"] # predictions: male or female
        print("\nNow we are going to see if income can be used to predict gender using a machine learning model for your selected region...")

    # ML data wrangling stuff
    # this section builds a neural network model to predict gender bsed on income. The goal is to see whether income alone can distinguish between male and female wage patterns. 
    # select input (X) and output(y)
    # X = region_data[["gross_income_euros"]] # feature: income values
    # y = region_data["gender"] # label: male or female
    # split the data into training and testing sets
    # 25% of data is used for testing
    # stratify=y keeps male and female proportions balanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100, stratify=y)

    # scale the data
    scaler = StandardScaler() # create scaler object
    scaler.fit(X_train) # learn scaling from training data only
    # apply scaling to both training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # create hte model
    model = MLPClassifier(max_iter=1000) # make neural network, give it up to 1000 tries to learn

    # train the model using training data
    model.fit(X_train_scaled, y_train)

    # evaluate accuracy on test data
    accuracy = model.score(X_test_scaled, y_test) # check how often model gets it right on the test data
    print("\nModel Accuracy: ", accuracy)

    # generate predictions for test data
    y_pred = model.predict(X_test_scaled)

    # create a confusion matrix (comparing actual vs predicted values)
    conf_matrix = confusion_matrix(y_test, y_pred) # compare real labels vs what the model predicted
    print("\nConfusion Matrix:\n", conf_matrix)

    # create labelled display of the confusion matrix
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)

    # plot the confusion matrix
    fig3, ax3 = plt.subplots() # create a new figure for the confusion matrix
    cm_disp.plot(ax=ax3, cmap="Pastel1") # use cmap for cute colors
    # title for chart depending on if it's about selected region or entire data set
    if second_choice == "A":
        ax3.set_title("Income → Gender Confusion Matrix for: Spain") # title for the chart
    else:
        ax3.set_title("Income → Gender Confusion Matrix for: " + selected_region) # title for the chart

    fig3.savefig("confusion_matrix.png") # save the confusion matrix image

    # classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
if __name__ == "__main__":
    main()

