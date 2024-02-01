import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("tidy_data.csv")
X = df.drop("Score", axis=1)
y = df["Score"]
X = df.drop(
    [
        "Time of day",
        "Quiz time",
        "name",
        "sunrise",
        "sunset",
        "moonphase",
        "conditions",
        "description",
        "icon",
        "stations",
        "Quiz author",
        "preciptype",
    ],
    axis=1,
)

points = df.copy()
points["ppp"] = points["Score"] / points["Participant Count"]
names = [
    "Allan",
    "Anna",
    "Ben",
    "Ben Karl",
    "Ben Kennerley",
    "Brett",
    "Casey Giberson",
    "Conor",
    "Daria",
    "David",
    "Dimitri",
    "Gary Stone",
    "Graeme",
    "Gudrun",
    "Harel",
    "Harriet Hall",
    "Isabelle",
    "Jaco",
    "James",
    "Jeannie",
    "Jill",
    "Jon",
    "Kabilan",
    "Keith",
    "Matt",
    "Maxim",
    "Nathan",
    "Nicole",
    "Paulse",
    "Perrie",
    "Prisca",
    "Quinn",
    "Ryan",
    "Sindiya",
    "Theo",
    "Virginie",
    "Yvonne",
]
person_count = points[names].sum()
for name in names:
    points[name] = points[name] * points["ppp"]

sum = points[names].sum()
sum_df = pd.DataFrame({"Name": sum.index, "Sum Value": sum.values})
sum_df["ppq"] = sum_df["Sum Value"] / person_count.values

# Sorting the DataFrame by 'Sum Value' in descending order
sum_df_sorted = sum_df.sort_values(by="Sum Value", ascending=False)

# Using seaborn to create a bar plot with ordered values and wider plot
plt.figure(figsize=(12, 6))  # Adjust the figsize for a wider plot
sns.barplot(x="Name", y="Sum Value", data=sum_df_sorted, palette="viridis")
plt.xlabel("Team Member")
plt.ylabel("Points Contributed")
plt.title("Who contributed the most?")
plt.xticks(rotation=45, ha="right")
plt.show()

sum_ppq_sorted = sum_df.sort_values(by="ppq", ascending=False)
# Using seaborn to create a bar plot with ordered values and wider plot
plt.figure(figsize=(12, 6))  # Adjust the figsize for a wider plot
sns.barplot(x="Name", y="ppq", data=sum_ppq_sorted, palette="viridis")
plt.xlabel("Team Member")
plt.ylabel("Points per quiz")
plt.title("Best value?")
plt.xticks(rotation=45, ha="right")
plt.show()


X = pd.get_dummies(X, columns=["Designated Guesser"], drop_first=True)
X = pd.get_dummies(X, columns=["1Q spinoff quiz"], drop_first=True)
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Creating and fitting a Lasso model
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)

# Making predictions
lasso_predictions = lasso.predict(X_test)

# Evaluate the model
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print(f"Lasso MSE: {lasso_mse}")
