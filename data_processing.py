import openpyxl
import pandas as pd

# read in csv file
df = pd.read_csv("quiz_scores.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)


# Function to find the last Thursday of the month
def last_thursday(date):
    next_month = date.month + 1
    next_year = date.year
    if next_month > 12:
        next_month = 1  # Reset to January of the next year
        next_year = date.year + 1  # Year value rolls over
    last_day = date.replace(day=1, month=next_month, year=next_year) - pd.DateOffset(
        days=1
    )
    return last_day - pd.DateOffset(days=(last_day.weekday() - 3) % 7)


# Function to find the third Wednesday of the month
def third_wednesday(date):
    first_day = date.replace(day=1)
    return (
        first_day
        + pd.DateOffset(days=(2 - first_day.weekday() + 7) % 7)
        + pd.DateOffset(weeks=2)
    )


# Function to find the first Wednesday of the month
def first_wednesday(date):
    first_day = date.replace(day=1)
    return first_day + pd.DateOffset(days=(2 - first_day.weekday() + 7) % 7)


# create list of meeting dates where food is given
meetings = []
for date in df["Date"]:
    briefing_date = last_thursday(date)
    meetings.append(briefing_date)
    data_date = third_wednesday(date)
    meetings.append(data_date)
    digital_date = first_wednesday(date)
    meetings.append(digital_date)

unique_meeting_dates = list(set(meetings))

# Create a dictionary to store the most recent meeting for each unique date
recent_meetings_dict = {
    date: max((value for value in unique_meeting_dates if value < date), default=None)
    for date in df["Date"]
}

# Calculate the number of days since the most recent meeting for each row
df["Days Since Free Food"] = df["Date"].apply(
    lambda date: (date - recent_meetings_dict[date]).days
    if date in recent_meetings_dict
    else None
)

# expand out participants list
df["Participants (number, office, charge out rate)"] = df[
    "Participants (number, office, charge out rate)"
].apply(lambda x: [item.strip() for item in x.split(";")])
df_expanded = df.explode("Participants (number, office, charge out rate)")

#  get counts, concatenate and make tidy
df_dummies = pd.get_dummies(
    df_expanded["Participants (number, office, charge out rate)"]
)
df_large = pd.concat([df_expanded, df_dummies], axis=1)
bool_columns = df_large.select_dtypes(include="bool").columns
df_names = df_large.groupby("Date")[bool_columns].sum().reset_index()
df_tidy = pd.merge(df, df_names, on="Date", how="left")

# drop irrelevant columns
df_tidy = pd.merge(df, df_names, on="Date", how="left")
cols_to_drop = [
    "Participants (number, office, charge out rate)",
    "TBC",
    "TBC with Keith",
    "",
    "Unnamed: 10",
    "Can't remember",
    "Can't remember",
]
df_tidy = df_tidy.drop(columns=cols_to_drop)

# add weather data
weather_df = pd.read_csv("full_weather_data.csv")
weather_df["datetime"] = pd.to_datetime(
    weather_df["datetime"], format="mixed", dayfirst=True
)
weather_df.rename(columns={"datetime": "Date"}, inplace=True)

df_tidy = pd.merge(df_tidy, weather_df, on="Date", how="left")
df_tidy = df_tidy.drop(columns="Date")
# add staff info (charge out rate, employee #)
staff_df = pd.read_excel("staff_info.xlsx")

# Initialize the total charge out rate column
df_tidy["Total_Charge_out_rate"] = 0

# Iterate over staff columns and merge data
for staff_column in df_tidy.columns[9:46]:
    # Extract staff number from column name
    matched_row = staff_df[staff_df["Name"].str.contains(staff_column, case=False)]
    if matched_row.empty:
        rate = 0
    else:
        rate = matched_row.iloc[0, 8]

    # Sum charge out rates for each meeting
    df_tidy["Total_Charge_out_rate"] += df_tidy[staff_column] * rate


# Adjust the first 3 values of '4 week rolling average' to remove NaN.
values = [11, 10.5, 11.33]
df_tidy["4 week rolling average"][0:3] = values

# Save the result
df_tidy.to_csv("tidy_data.csv", index=False)
