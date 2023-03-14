#########################################################################
#     Rule Based Classification for Calculation Potential Customers
#########################################################################

# Import Packages and dataset
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("datasets/persona.csv")
df.head()
df.info()


#################################################
# General Table of Dataset and Defining Functions
#################################################
def check_df(dataframe, head=5):
    print("################ Shape ##################")
    print(dataframe.shape)
    print("################ Types ##################")
    print(dataframe.dtypes)
    print("################ Head ##################")
    print(dataframe.head(head))
    print("################ Tail ##################")
    print(dataframe.tail(head))
    print("################ NA ##################")
    if dataframe.isnull().values.any():
        print(dataframe.isnull().sum())
    else:
        print("There is no NA")
    print("################ Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("################ Value Counts for Each Column ##################")
    for col in dataframe.columns:
        if dataframe[col].nunique() > 10:
            print("Too many elements for *{}*".format(col))
            continue
        else:
            print(f"{col}: {dataframe[col].value_counts()}")


def categorical_ratio(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        'Ratio': 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##############################")
    print(f"Total unique value is: {dataframe[col_name].nunique()}")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# Answers for General Table According to check.df Func. and some basic codes.
# ----------------------------------------------------------------------------
# How many unique "SOURCE"? ---> 2 SOURCE: ["android", "ios"]
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
# How many unique "PRICE"? ---> 6 PRICE: [39 49 29 19 59  9]
# How many sales were made from which "PRICE"?
check_df(df)
# How many sales from which country?
check_df(df)
# What is the total profit according to Countries?
categorical_ratio(df, "COUNTRY", plot=True)
df.groupby("COUNTRY").agg({"PRICE": ["sum", "mean"]})
# What is the sales numbers according to "SOURCE"?
categorical_ratio(df, "SOURCE", plot=True)
# What are the "PRICE" averages according to countries?
df.groupby("COUNTRY").agg({"PRICE": ["sum", "mean"]})
# What are the "SOURCE" averages according to countries?
df.groupby("SOURCE").agg({"PRICE": ["sum", "mean"]})
# What are the "PRICE" averages by COUNTRY-SOURCE?
df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE": ["sum", "mean"]})

# Average Earnings in Breakdown of "COUNTRY", "SOURCE", "SEX", "AGE"
# Ascending=False and assign the agg_df for output
# -------------------------------------------------------------------
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
agg_df.sort_values("PRICE", ascending=False)

# Reset the index of agg_df
agg_df = agg_df.reset_index()
print(agg_df)

# Change the "AGE" variable type to categorical.
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70],
                           labels=["0_18", "19_23", "24_30", "31_40", "41_70"])

# Creating a new level based customers (persona) and adding dataset as a variable.
print(agg_df.values)
agg_df["customer_level_based"] = [row_index[0].upper() + "_" +
                                  row_index[1].upper() + "_" +
                                  row_index[2].upper() + "_" +
                                  row_index[5].upper()
                                  for row_index in agg_df.values]
agg_df.head()

# To prevent multiple identical statements.
agg_df = agg_df.groupby("customer_level_based")["PRICE"].mean().reset_index()
agg_df["customer_level_based"].value_counts()
agg_df.head()

# Creating segments for new customers.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, ["D", "C", "B", "A"])
agg_df.head(20)

check_df(agg_df)
for col in agg_df.columns:
    categorical_ratio(df, col)

# Adding new customers to dataset
new_user = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customer_level_based"] == new_user])


def predictor_by_agg_df(country, source, sex, age, col_name="customer_level_based", dataframe=agg_df):
    # Changing age to string
    if int(age) <= 18:
        age = "0_18"
    if 19 <= int(age) <= 23:
        age = "19_23"
    if 24 <= int(age) <= 30:
        age = "24_30"
    if 31 <= int(age) <= 40:
        age = "31_40"
    else:
        age = "41_70"
    string_clb = country + "_" + source + "_" + sex + "_" + age
    print(dataframe[dataframe[col_name] == string_clb])


# Basic User Command:
print("******************************")
print(df["COUNTRY"].unique())
print(df["SOURCE"].unique())
print(df["SEX"].unique())
print("******************************")
print("Please select your values according to above lists.")
input_country = input("Please select country:")
input_country = input_country.upper()
input_source = input("Please select source:")
input_source = input_source.upper()
input_sex = input("Please select sex:")
input_sex = input_sex.upper()
input_age = input("Please enter an age:")

predictor_by_agg_df(input_country, input_source, input_sex, input_age)
