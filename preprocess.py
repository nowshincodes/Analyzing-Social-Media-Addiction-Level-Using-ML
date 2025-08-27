import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


df = pd.read_csv(r"V:\CSE424 project\Social-media.csv")


encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['Gender','Dominant_Emotion']])


encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['Gender','Dominant_Emotion']))

df = df.drop(columns=['Gender','Dominant_Emotion','Platform'])


df_combined = pd.concat([df, encoded_df], axis=1)

# df_combined['Engagement_Score'] = (
#     0.3 * df_combined['Likes_Received_Per_Day'] +
#     0.4 * df_combined['Comments_Received_Per_Day'] +
#     0.6 * df_combined['Posts_Per_Day']+
#     0.7 * df_combined['Messages_Sent_Per_Day'] 
    

# )

Dominant_Emotion_weights = {
    'Dominant_Emotion_Happiness': 1.0,
    'Dominant_Emotion_Neutral': 0.0,
    'Dominant_Emotion_Boredom': -0.3,
    'Dominant_Emotion_Sadness': -0.6,
    'Dominant_Emotion_Anxiety': -0.7,
    'Dominant_Emotion_Anger': -0.8,
    'Dominant_Emotion_Agression': -1.0
}

df_combined['Emotional_Polarity'] = sum(df_combined[col] * weight for col, weight in Dominant_Emotion_weights.items())
df_combined = df_combined.drop(columns=[    'Dominant_Emotion_Happiness',
    'Dominant_Emotion_Neutral',
    'Dominant_Emotion_Boredom',
    'Dominant_Emotion_Sadness',
    'Dominant_Emotion_Anxiety',
    'Dominant_Emotion_Anger',
    'Dominant_Emotion_Agression'

    ]
    )

#visualization
corr_matrix = df_combined.corr(numeric_only=True) 

# Step 3: Create the heatmap
plt.figure(figsize=(10, 8))  # optional: adjust size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Step 4: Show the plot
plt.title("Correlation Heatmap")
plt.show()


# Calculation for addiction level using PCA

X = df_combined[['Daily_Usage_Time (minutes)',    'Likes_Received_Per_Day',
    'Comments_Received_Per_Day',
    "Posts_Per_Day",
    'Messages_Sent_Per_Day']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=1)  # Extract 1 component for addiction level
pc1_scores = pca.fit_transform(X_scaled)  # PC1 scores for each user

# Step 3: Interpret results
print("Explained variance ratio of PC1:", pca.explained_variance_ratio_[0])
print("Loadings for PC1:", pca.components_[0])

# Step 4: Add PC1 scores to the dataset
df_combined['Addiction_Level'] = pc1_scores

# Step 5: Classify (example with percentiles)
thresholds = np.percentile(pc1_scores, [40, 70])
df_combined['Addiction_Class'] = pd.cut(
    df_combined['Addiction_Level'],
    bins=[-float('inf'), thresholds[0], thresholds[1], float('inf')],
    labels=['Low', 'Medium', 'High']
)

# Calculation for productivity level using PCA
X = df_combined[['Addiction_Level',    'Emotional_Polarity']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=1)  # Extract 1 component for productive
pc1_scores = pca.fit_transform(X_scaled)  # PC1 scores for each user

# Step 3: Interpret results
print("Explained variance ratio of PC1:", pca.explained_variance_ratio_[0])
print("Loadings for PC1:", pca.components_[0])

correlation = df_combined['Addiction_Level'].corr(df_combined['Emotional_Polarity'])
print("Correlation between Feature1 and Feature2:", correlation)

# Step 4: Add PC1 scores to the dataset
df_combined['Productivity_score'] = pc1_scores
df_combined['Productivity_score'] = df_combined['Productivity_score']

df_combined.info()
print(df_combined.head(20))

category_counts = df_combined['Addiction_Class'].value_counts()

# Step 3: Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)

# Step 4: Display the chart
plt.title("Distribution of Classification Target")
plt.axis('equal')  # Ensures pie is a circle
plt.show()