import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('vgsales.csv')

X = df[['EU_Sales', 'NA_Sales']].values
y = df['Genre'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the testing dataset
y_pred = gnb.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Separate the correctly classified and incorrectly classified points
correctly_classified = (y_pred == y_test)
incorrectly_classified = (y_pred != y_test)

# Plot the incorrectly classified points in green
plt.scatter(X_test[correctly_classified, 0], X_test[correctly_classified, 1], c='green', label='Correctly Classified')

# Plot the incorrectly classified points in red
plt.scatter(X_test[incorrectly_classified, 0], X_test[incorrectly_classified, 1], c='red', label='Incorrectly Classified')


plt.xlabel('EU Sales')
plt.ylabel('NA Sales')
plt.title('Predicted Game Genres')
plt.show()
