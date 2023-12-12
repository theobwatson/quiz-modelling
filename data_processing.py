import pandas as pd

df = pd.read_csv('quiz_scores.csv')
df['Participants (number, office, charge out rate)'] = df['Participants (number, office, charge out rate)'].apply(lambda x: x.split(';'))
df_expanded = df.explode('Participants (number, office, charge out rate)')

df_dummies = pd.get_dummies(df_expanded['Participants (number, office, charge out rate)'])
df = pd.concat([df_expanded, df_dummies], axis=1)
df_tidy = df.groupby('Date').agg('sum').reset_index() # incorrectly sums all attributes
df_tidy = df_tidy.fillna(0)

