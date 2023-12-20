import openpyxl
import pandas as pd

# read in csv file
df = pd.read_csv('quiz_scores.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# expand out participants list
df['Participants (number, office, charge out rate)'] = df['Participants (number, office, charge out rate)'].apply(lambda x: [item.strip() for item in x.split(';')])
df_expanded = df.explode('Participants (number, office, charge out rate)')

#  get counts, concatenate and make tidy
df_dummies = pd.get_dummies(df_expanded['Participants (number, office, charge out rate)'])
df_large = pd.concat([df_expanded, df_dummies], axis=1)
bool_columns = df_large.select_dtypes(include='bool').columns
df_names = df_large.groupby('Date')[bool_columns].sum().reset_index()
df_tidy = pd.merge(df, df_names, on='Date', how='left')

# drop irrelevant columns
df_tidy = pd.merge(df, df_names, on='Date', how='left')
cols_to_drop = ['Participants (number, office, charge out rate)', 'TBC', 'TBC with Keith', '', 'Unnamed: 10', "Can't remember", "Can't remember"]
df_tidy = df_tidy.drop(columns=cols_to_drop)

# add weather data
weather_df = pd.read_csv('full_weather_data.csv')
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], format='mixed',dayfirst=True)
weather_df.rename(columns={'datetime': 'Date'}, inplace=True)

df_tidy = pd.merge(df_tidy, weather_df, on='Date', how='left')
df_tidy = df_tidy.drop(columns='Date')
# add staff info (charge out rate, employee #)
staff_df = pd.read_excel('staff_info.xlsx')

# Initialize the total charge out rate column
df_tidy['Total_Charge_out_rate'] = 0

# Iterate over staff columns and merge data
for staff_column in df_tidy.columns[8:45]:
    # Extract staff number from column name
    matched_row = staff_df[staff_df['Name'].str.contains(staff_column, case=False)]
    if matched_row.empty:
        rate = 0
    else:
        rate = matched_row.iloc[0, 8]

    # Sum charge out rates for each meeting
    df_tidy['Total_Charge_out_rate'] += df_tidy[staff_column]*rate

# Save the result
df_tidy.to_csv('tidy_data.csv', index=False)




