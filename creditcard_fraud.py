#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import warnings


# In[2]:


# Load the data from the Excel file
data = pd.read_excel('credit_fraud_detection_dataset.xlsx')


# In[3]:


import os

# Load the data from the Excel file
data = pd.read_excel('credit_fraud_detection_dataset.xlsx')

# List of attributes to cluster
attributes = [
    'Age', 'Income', 'Credit Score', 'Number_of_Existing_Credit_Cards',
    'Number_of_Existing_Bank_Accounts', 'Number of Social Media Connections',
    'Social Media Account Age'
]

# Helper function to apply K-means clustering and visualize the results
def cluster_and_visualize(data, column_name, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data[f'{column_name}_cluster'] = kmeans.fit_predict(data[[column_name]].dropna())

    # Visualize the clusters with distinct colors
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette('viridis', n_clusters)
    sns.histplot(data=data, x=column_name, hue=f'{column_name}_cluster', palette=palette, bins=30)

    # Add custom legend
    handles = [plt.Line2D([0], [0], color=palette[i], lw=4) for i in range(n_clusters)]
    labels = [f'Cluster {i}' for i in range(n_clusters)]
    plt.legend(handles=handles, labels=labels, title='Cluster')

    plt.title(f'{column_name} Distribution by Cluster')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

# Apply clustering and visualize for each attribute
for attribute in attributes:
    cluster_and_visualize(data, attribute)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set the environment variable to avoid memory leak issue with KMeans on Windows
os.environ['OMP_NUM_THREADS'] = '2'


# #### Income:
# Cluster 0: Lower income (range $10,000-$40,000) - Higher risk due to financial pressures and limited resources.
# 
# Cluster 1: Middle income (range $41,000-$90,000) - Lower risk due to financial stability.
# 
# Cluster 2: Higher income (range $91,000-$150,000) - Lower risk due to substantial financial resources.
# 
# #### Credit Score:
# Cluster 0: Good credit (range 750-850) - Lower risk indicating strong credit history and low financial risk.
# 
# Cluster 1: Fair credit (range 650-749) - Moderate risk indicating average financial reliability.
# 
# Cluster 2: Poor credit (range 300-649) - Higher risk indicating poor credit history and financial instability.
# 
# #### Number of Existing Credit Cards:
# Cluster 0: Few credit cards (range 0-2) - Lower risk due to limited credit exposure.
# 
# Cluster 1: Moderate number of credit cards (range 3-5) - Moderate risk due to balanced credit usage.
# 
# Cluster 2: Many credit cards (range 6-10) - Higher risk due to higher credit exposure and potential financial mismanagement.
# 
# #### Number of Social Media Connections:
# Cluster 0: Few connections (range 0-100) - Lower risk due to limited social media exposure.
# 
# Cluster 1: Moderate connections (range 101-200) - Moderate risk due to balanced social media presence.
# 
# Cluster 2: Many connections (range 201-500) - Higher risk due to higher social media exposure and potential security risks.
# 
# #### Social Media Account Age:
# Cluster 0: New accounts (range 0-2 years) - Higher risk due to recent account creation and lack of history.
# 
# Cluster 1: Moderately aged accounts (range 3-6 years) - Moderate risk due to balanced account history.
# 
# Cluster 2: Old accounts (range 7-15 years) - Lower risk due to long account history and established presence.
# 
# #### Age:
# Cluster 0: Middle age (range 36-45) - Lower risk due to financial stability and experience.
# 
# Cluster 1: Older age (range 45-70) - Moderate risk due to potential vulnerabilities related to age.
# 
# Cluster 2: Young age (range 18-35) - Higher risk due to lower financial literacy and fewer financial responsibilities.
# 
# 
# #### Social Media Account Age:
# Cluster 0: New account (<1 year) - Higher risk as new accounts might be created for fraudulent purposes.
# 
# Cluster 1: Established account (1-5 years) - Moderate risk.
# 
# Cluster 2: Long-term account (>5 years) - Lower risk indicating responsible behavior.
# 
# #### State/City:
# Cluster 0: Major metropolitan areas - Higher risk due to population density.
# 
# Cluster 1: Suburban areas - Moderate risk.
# 
# Cluster 2: Rural areas - Lower risk.
# 
# #### Gender:
# Cluster 0: Male - Equal risk as female.
# 
# Cluster 1: Female - Equal risk as male.
# 
# #### Education Level:
# Cluster 0: High school or less - Higher risk due to less awareness.
# 
# Cluster 1: Bachelor’s degree - Moderate risk.
# 
# Cluster 2: Master’s degree - Lower risk.
# 
# Cluster 3: PhD - Lowest risk.
# 
# 
# 
# #### Employment Status:
# Cluster 0: Unemployed - Higher risk due to financial desperation.
# 
# Cluster 1: Part-time - Moderate risk.
# 
# Cluster 2: Full-time - Lower risk.
# 
# #### Phone Number Verified:
# Cluster 0: Not verified - Higher risk.
# 
# Cluster 1: Verified - Lower risk.
# 
# #### Email Verified:
# Cluster 0: Not verified - Higher risk.
# 
# Cluster 1: Verified - Lower risk.
# 
# #### SSN Number Verified:
# Cluster 0: Not verified - Higher risk.
# 
# Cluster 1: Verified - Lower risk.
# 
# #### Social Media Presence:
# Cluster 0: Low presence - Lower risk.
# 
# Cluster 1: Moderate presence - Moderate risk.
# 
# Cluster 2: High presence - Higher risk.
# 
# #### Browser Type:
# Cluster 0: Common browsers (e.g., Chrome, Firefox) - Lower risk.
# 
# Cluster 1: Less common browsers (e.g., Safari, Edge) - Moderate risk.
# 
# Cluster 2: Rare browsers - Higher risk.
# 
# #### Operating System:
# Cluster 0: Windows - Lower risk.
# 
# Cluster 1: MacOS - Moderate risk.
# 
# Cluster 2: Other (Linux, etc.) - Higher risk.
# 
# #### Mouse Movement:
# Cluster 0: Minimal movement - Lower risk.
# 
# Cluster 1: Average movement - Moderate risk.
# 
# Cluster 2: High movement - Higher risk.
# 
# #### Click Pattern:
# Cluster 0: Rare clicks - Lower risk.
# 
# Cluster 1: Moderate clicks - Moderate risk.
# 
# Cluster 2: Frequent clicks - Higher risk.
# 
# #### Device Location:
# Cluster 0: Home - Lower risk.
# 
# Cluster 1: Work - Moderate risk.
# 
# Cluster 2: Public places (cafes, libraries, etc.) - Higher risk.
# 
# #### Typing Speed:
# Cluster 0: Slow (<30 WPM) - Lower risk.
# 
# Cluster 1: Average (30-60 WPM) - Moderate risk.
# 
# Cluster 2: Fast (>60 WPM) - Higher risk.
# 
# #### Scroll Speed:
# Cluster 0: Slow - Lower risk.
# 
# Cluster 1: Average - Moderate risk.
# 
# Cluster 2: Fast - Higher risk.

# In[4]:


# List of categorical attributes to cluster
from sklearn.preprocessing import LabelEncoder
categorical_attributes = [
    'State', 'City', 'Gender', 'Education Level', 'Employment Status',
    'Phone_Number_Verified', 'Email_Verified', 'SSN_Number_Verified',
    'Social Media Presence', 'Browser Type', 'Operating System',
    'Mouse Movement', 'Click Pattern', 'Device Location','Typing Speed','Scroll Speed'

]

# Helper function to encode categorical data, apply K-means clustering, and visualize the results
def cluster_and_visualize_categorical(data, column_name, n_clusters=3):
    # Encode categorical data
    le = LabelEncoder()
    data[f'{column_name}_encoded'] = le.fit_transform(data[column_name].astype(str))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data[f'{column_name}_cluster'] = kmeans.fit_predict(data[[f'{column_name}_encoded']].dropna())

    # Visualize the clusters
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=column_name, hue=f'{column_name}_cluster', palette='viridis')
    plt.title(f'{column_name} Distribution by Cluster')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.legend(title='Cluster')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Apply clustering and visualize for each categorical attribute
for attribute in categorical_attributes:
    cluster_and_visualize_categorical(data, attribute)


# In[5]:


# Assign initial scores for clusters

cluster_scores = {
    'Income': {0: 50, 1: 30, 2: 20},  # Higher risk with lower income
    'Credit Score': {0: 20, 1: 30, 2: 50},  # Higher risk with lower credit score
    'Number_of_Existing_Credit_Cards': {0: 20, 1: 30, 2: 50},  # Higher risk with more credit cards
    'Age': {0: 30, 1: 20, 2: 50},  # Higher risk for younger and older individuals
    'Number_of_Existing_Bank_Accounts': {0: 20, 1: 30, 2: 50},  # Higher risk with more bank accounts
    'Number of Social Media Connections': {0: 20, 1: 30, 2: 50},  # Higher risk with more connections
    'Social Media Account Age': {0: 50, 1: 30, 2: 20}  # Higher risk with new accounts
}


# In[6]:


def map_data(data):
    # Simulate mapping string categories to the expected cluster keys
    data['State_cluster'] = data['State'].map({'Metropolitan': 0, 'Suburban': 1, 'Rural': 2})
    data['City_cluster'] = data['City'].map({'Metropolitan': 0, 'Suburban': 1, 'Rural': 2})
    data['Gender_cluster'] = data['Gender'].map({'Male': 0, 'Female': 1})
    data['Education_Level_cluster'] = data['Education Level'].map({'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3})
    data['Employment_Status_cluster'] = data['Employment Status'].map({'Unemployed': 0, 'Part-time': 1, 'Full-time': 2})
    data['Phone_Number_Verified_cluster'] = data['Phone_Number_Verified'].map({'No': 0, 'Yes': 1})
    data['Email_Verified_cluster'] = data['Email_Verified'].map({'No': 0, 'Yes': 1})
    data['SSN_Number_Verified_cluster'] = data['SSN_Number_Verified'].map({'No': 0, 'Yes': 1})
    data['Social_Media_Presence_cluster'] = data['Social Media Presence'].map({'Low': 0, 'Moderate': 1, 'High': 2})
    data['Browser_Type_cluster'] = data['Browser Type'].map({'Common': 0, 'Less Common': 1, 'Rare': 2})
    data['Operating_System_cluster'] = data['Operating System'].map({'Windows': 0, 'MacOS': 1, 'Other': 2})
    data['Mouse_Movement_cluster'] = data['Mouse Movement'].map({'Minimal': 0, 'Average': 1, 'High': 2})
    data['Click_Pattern_cluster'] = data['Click Pattern'].map({'Rare': 0, 'Moderate': 1, 'Frequent': 2})
    data['Device_Location_cluster'] = data['Device Location'].map({'Home': 0, 'Work': 1, 'Public': 2})
    data['Typing_Speed_cluster'] = data['Typing Speed'].map({'Slow': 0, 'Average': 1, 'Fast': 2})
    data['Scroll_Speed_cluster'] = data['Scroll Speed'].map({'Slow': 0, 'Average': 1, 'Fast': 2})
    return data


# In[7]:


cluster_scores = {
    'State': {0: 50, 1: 30, 2: 20},
    'City': {0: 50, 1: 30, 2: 20},
    'Gender': {0: 50, 1: 50},
    'Education Level': {0: 50, 1: 30, 2: 15, 3: 5},
    'Employment Status': {0: 50, 1: 30, 2: 20},
    'Phone_Number_Verified': {0: 75, 1: 25},
    'Email_Verified': {0: 75, 1: 25},
    'SSN_Number_Verified': {0: 75, 1: 25},
    'Social Media Presence': {0: 20, 1: 30, 2: 50},
    'Browser Type': {0: 20, 1: 30, 2: 50},
    'Operating System': {0: 20, 1: 30, 2: 50},
    'Mouse Movement': {0: 20, 1: 30, 2: 50},
    'Click Pattern': {0: 20, 1: 30, 2: 50},
    'Device Location': {0: 20, 1: 30, 2: 50},
    'Typing Speed': {0: 20, 1: 30, 2: 50},
    'Scroll Speed': {0: 20, 1: 30, 2: 50}
}


# In[8]:


# Define weightages for attributes based on their potential impact on fraud detection
attribute_weightages = {
    'Age': 0.1,
    'Income': 0.15,
    'Credit Score': 0.15,
    'Number_of_Existing_Credit_Cards': 0.1,
    'Number_of_Existing_Bank_Accounts': 0.1,
    'Number of Social Media Connections': 0.05,
    'Social Media Account Age': 0.05,
    'State': 0.05,
    'City': 0.05,
    'Gender': 0.05,
    'Education Level': 0.05,
    'Employment Status': 0.05,
    'Phone_Number_Verified': 0.025,
    'Email_Verified': 0.025,
    'SSN_Number_Verified': 0.025,
    'Social Media Presence': 0.05,
    'Browser Type': 0.05,
    'Operating System': 0.05,
    'Mouse Movement': 0.05,
    'Click Pattern': 0.05,
    'Device Location': 0.05,
    'Typing Speed': 0.05,
    'Scroll Speed': 0.05
}


# In[9]:


def calculate_scaled_risk_scores(data, cluster_scores, attribute_weightages):
    risk_scores = pd.DataFrame()
    for attribute, scores in cluster_scores.items():
        cluster_col = f'{attribute}_cluster'
        if cluster_col in data.columns:
            data[f'{attribute}_risk_score'] = data[cluster_col].map(scores).fillna(0)
            risk_scores[f'{attribute}_risk_score'] = data[f'{attribute}_risk_score'] * attribute_weightages[attribute]
        else:
            risk_scores[f'{attribute}_risk_score'] = 0  # Default to 0 if no data available

    # Combine and scale risk scores
    risk_scores['total_risk_score'] = risk_scores.sum(axis=1)
    scaler = MinMaxScaler(feature_range=(350, 1000))
    risk_scores['scaled_risk_score'] = scaler.fit_transform(risk_scores[['total_risk_score']])

    return risk_scores


# In[10]:


# Calculate and display risk scores
from sklearn.preprocessing import MinMaxScaler
risk_scores = calculate_scaled_risk_scores(data, cluster_scores, attribute_weightages)
print(risk_scores.head())


# In[11]:


# Add decision suggestions based on risk scores
def decision_suggestions(risk_scores):
    conditions = [
        (risk_scores['scaled_risk_score'] <= 600),
        (risk_scores['scaled_risk_score'] > 600) & (risk_scores['scaled_risk_score'] <= 800),
        (risk_scores['scaled_risk_score'] > 800)
    ]
    choices = ['Accepted', 'Suspicious', 'Declined']
    risk_scores['decision'] = np.select(conditions, choices, default='Review')

    return risk_scores

# Apply decision suggestions
final_scores = decision_suggestions(risk_scores)
final_scores.head()


# In[12]:


def calculate_scaled_risk_scores(data, cluster_scores, attribute_weightages):
    risk_scores = pd.DataFrame()
    for attribute, scores in cluster_scores.items():
        cluster_col = f'{attribute}_cluster'
        if cluster_col in data.columns:
            data[f'{attribute}_risk_score'] = data[cluster_col].map(scores).fillna(0)
            risk_scores[f'{attribute}_risk_score'] = data[f'{attribute}_risk_score'] * attribute_weightages[attribute]
        else:
            risk_scores[f'{attribute}_risk_score'] = 0  # Default to 0 if no data available

    # Combine and scale risk scores
    risk_scores['total_risk_score'] = risk_scores.sum(axis=1)
    scaler = MinMaxScaler(feature_range=(350, 1000))
    risk_scores['scaled_risk_score'] = scaler.fit_transform(risk_scores[['total_risk_score']])

    return risk_scores



# Calculate risk scores
risk_scores = calculate_scaled_risk_scores(data, cluster_scores, attribute_weightages)

# Concatenate original data with risk scores
result = pd.concat([data, risk_scores], axis=1)

# Add a decision column based on the scaled risk score
result['decision'] = result['scaled_risk_score'].apply(lambda x: 'Accepted' if x < 600 else ('Suspicious' if x < 800 else 'Rejected'))

# Display the first few rows of the result
print(result.head())


# In[13]:


result_df = pd.DataFrame(result)


# In[14]:


result_df


# In[15]:


pip install gradio


# In[16]:


import pandas as pd
import numpy as np
import gradio as gr
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import base64


# Define attributes for clustering
attributes = [
    'Age', 'Income', 'Credit Score', 'Number_of_Existing_Credit_Cards',
    'Number_of_Existing_Bank_Accounts', 'Number of Social Media Connections',
    'Social Media Account Age'
]

categorical_attributes = [
    'State', 'City', 'Gender', 'Education Level', 'Employment Status',
    'Phone_Number_Verified', 'Email_Verified', 'SSN_Number_Verified',
    'Social Media Presence', 'Browser Type', 'Operating System',
    'Mouse Movement', 'Click Pattern', 'Device Location', 'Typing Speed', 'Scroll Speed'
]

# Preprocess the data
def preprocess_data(data, attributes, categorical_attributes):
    # Fill missing values for numeric attributes
    for col in attributes:
        if data[col].dtype in ['int64', 'float64']:
            data[col].fillna(data[col].median(), inplace=True)

    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_attributes:
        data[col] = le.fit_transform(data[col].astype(str))

    return data

data = preprocess_data(data, attributes, categorical_attributes)

# Apply K-means clustering to numerical attributes and save the cluster centroids
def apply_clustering(data, attributes, n_clusters=3):
    centroids = {}
    for attribute in attributes:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data[f'{attribute}_cluster'] = kmeans.fit_predict(data[[attribute]].dropna())
        centroids[attribute] = kmeans.cluster_centers_
    return data, centroids

data, centroids = apply_clustering(data, attributes)

# Assign initial scores for clusters
cluster_scores = {
    'Income': {0: 50, 1: 30, 2: 20},  # Higher risk with lower income
    'Credit Score': {0: 20, 1: 30, 2: 50},  # Higher risk with lower credit score
    'Number_of_Existing_Credit_Cards': {0: 20, 1: 30, 2: 50},  # Higher risk with more credit cards
    'Age': {0: 50, 1: 20, 2: 30},  # Higher risk for younger and older individuals
    'Number_of_Existing_Bank_Accounts': {0: 20, 1: 30, 2: 50},  # Higher risk with more bank accounts
    'Number of Social Media Connections': {0: 20, 1: 30, 2: 50},  # Higher risk with more connections
    'Social Media Account Age': {0: 50, 1: 30, 2: 20}  # Higher risk with new accounts
}

# Define weightages for attributes based on their potential impact on fraud detection
attribute_weightages = {
    'Age': 0.1,
    'Income': 0.15,
    'Credit Score': 0.15,
    'Number_of_Existing_Credit_Cards': 0.1,
    'Number_of_Existing_Bank_Accounts': 0.1,
    'Number of Social Media Connections': 0.05,
    'Social Media Account Age': 0.05,
    'State': 0.05,
    'City': 0.05,
    'Gender': 0.05,
    'Education Level': 0.05,
    'Employment Status': 0.05,
    'Phone_Number_Verified': 0.025,
    'Email_Verified': 0.025,
    'SSN_Number_Verified': 0.025,
    'Social Media Presence': 0.05,
    'Browser Type': 0.05,
    'Operating System': 0.05,
    'Mouse Movement': 0.05,
    'Click Pattern': 0.05,
    'Device Location': 0.05,
    'Typing Speed': 0.05,
    'Scroll Speed': 0.05
}

# Calculate scaled risk scores
def calculate_scaled_risk_scores(data, cluster_scores, attribute_weightages):
    risk_scores = pd.DataFrame()
    for attribute, scores in cluster_scores.items():
        cluster_col = f'{attribute}_cluster'
        if cluster_col in data.columns:
            data[f'{attribute}_risk_score'] = data[cluster_col].map(scores).fillna(0)
            risk_scores[f'{attribute}_risk_score'] = data[f'{attribute}_risk_score'] * attribute_weightages[attribute]
        else:
            risk_scores[f'{attribute}_risk_score'] = 0  # Default to 0 if no data available

    # Combine and scale risk scores
    risk_scores['total_risk_score'] = risk_scores.sum(axis=1)
    scaler = MinMaxScaler(feature_range=(350, 1000))
    risk_scores['scaled_risk_score'] = scaler.fit_transform(risk_scores[['total_risk_score']])

    return risk_scores, scaler

# Calculate risk scores and fit scaler on the entire dataset
risk_scores, scaler = calculate_scaled_risk_scores(data, cluster_scores, attribute_weightages)

# Concatenate original data with risk scores
result = pd.concat([data, risk_scores], axis=1)

# Add a decision column based on the scaled risk score
result['decision'] = result['scaled_risk_score'].apply(lambda x: 'Accepted' if x < 600 else ('Suspicious' if x < 800 else 'Rejected'))

# Display the first few rows of the result
print(result.head())

# Function to assign clusters to a new sample based on the nearest centroid
def assign_clusters(sample, centroids):
    clusters = {}
    for attribute, centroid in centroids.items():
        sample_value = sample[attribute].values[0]
        distances = np.linalg.norm(centroid - sample_value, axis=1)
        clusters[f'{attribute}_cluster'] = np.argmin(distances)
    return clusters

# Function to process the new input and predict the risk score
def predict_risk(*args):
    sample = dict(zip(attributes + categorical_attributes, args))
    sample_df = pd.DataFrame([sample])

    # Assign clusters to the sample
    sample_clusters = assign_clusters(sample_df, centroids)
    for col, cluster in sample_clusters.items():
        sample_df[col] = cluster

    # Calculate risk scores for the sample
    sample_risk_scores, _ = calculate_scaled_risk_scores(sample_df, cluster_scores, attribute_weightages)
    sample_risk_scores['scaled_risk_score'] = scaler.transform(sample_risk_scores[['total_risk_score']])
    sample_result = pd.concat([sample_df, sample_risk_scores], axis=1)
    sample_result['decision'] = sample_result['scaled_risk_score'].apply(lambda x: 'Accepted' if x < 600 else ('Suspicious' if x < 800 else 'Rejected'))

    return sample_result['scaled_risk_score'].values[0], sample_result['decision'].values[0]

# Create Gradio interface
numeric_inputs = [gr.Number(label=attr, value=data[attr].median()) for attr in attributes if data[attr].dtype in ['int64', 'float64']]
categorical_inputs = [gr.Dropdown(choices=data[attr].unique().tolist(), label=attr) for attr in categorical_attributes]

css = """
body {
    background: url('https://www2.deloitte.com/content/dam/Deloitte/global/Images/promo_images/Backgrounds/gx-bg-deloitte-wave.jpg') no-repeat center center fixed;
    background-size: cover;
    color: white;
    font-family: "Deloitte", sans-serif;
}
"""

logo_path = 'download.jpg'

with open(logo_path, "rb") as f:
    logo_data = f.read()
    logo_base64 = base64.b64encode(logo_data).decode('utf-8')

html_header = f"""
<div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
    <img src="data:image/png;base64,{logo_base64}" alt="Deloitte Logo" style="height: 100px;"/>
</div>
"""

# Gradio Interface
header = gr.HTML(html_header)
iface = gr.Interface(
    fn=predict_risk,
    inputs=numeric_inputs + categorical_inputs,
    outputs=[gr.Textbox(label="Scaled Risk Score"), gr.Textbox(label="Decision")],
    title="Deloitte Fraud Risk Prediction",
    description="Enter the details to predict the risk score and decision.",
    css=css,
    theme="default"
)

# Combine Header and Interface
with gr.Blocks() as demo:
    header.render()
    iface.render()

demo.launch()


# In[ ]:




