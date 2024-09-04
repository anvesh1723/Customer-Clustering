import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

# Sidebar
st.sidebar.title("Customer Clustering App")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Display first 5 rows of the dataframe
if uploaded_file is not None:
    customer_data = pd.read_csv(uploaded_file)
    st.subheader("Customer Data")
    st.write(customer_data.head())

    # Data Analysis
    st.sidebar.subheader("Data Analysis")

    # Show dataset info
    if st.sidebar.checkbox("Show Dataset Info"):
        st.write("Number of rows and columns:", customer_data.shape)
        st.write("Dataset Info:")
        st.write(customer_data.info())

    # Show missing values
    if st.sidebar.checkbox("Show Missing Values"):
        st.write("Missing Values:")
        st.write(customer_data.isnull().sum())

    # Choose columns
    st.sidebar.subheader("Choose Columns")
    selected_columns = st.sidebar.multiselect("Select Columns:", customer_data.columns)

    # Display selected columns
    if selected_columns:
        st.subheader("Selected Columns:")
        st.write(customer_data[selected_columns])

    # Clustering
    st.sidebar.subheader("Clustering")

    # Choose clustering algorithm
    clustering_algorithm = st.sidebar.selectbox("Select Clustering Algorithm:", ["K-Means", "DBSCAN"])

    if clustering_algorithm == "K-Means":
        st.subheader("K-Means Clustering")

        # Choose features
        feature1 = st.sidebar.selectbox("Select Feature 1:", customer_data.columns)
        feature2 = st.sidebar.selectbox("Select Feature 2:", customer_data.columns)

        # Choose number of clusters
        n_clusters = st.sidebar.slider("Select Number of Clusters:", min_value=2, max_value=10, value=5)

        # Extract features for clustering
        X = customer_data[[feature1, feature2]].values

        # Plot Elbow Point Graph
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        st.subheader("Elbow Point Graph")
        st.line_chart(pd.DataFrame({"Number of Clusters": range(1, 11), "WCSS": wcss}).set_index("Number of Clusters"))

        # Train K-Means model
        kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
        Y = kmeans_model.fit_predict(X)

        # Plot Clusters
        st.subheader("Customer Groups")
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis')
        plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
        plt.title('Customer Groups')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        st.pyplot(plt)

    elif clustering_algorithm == "DBSCAN":
        st.subheader("DBSCAN Clustering")

        # Choose features
        feature1 = st.sidebar.selectbox("Select Feature 1:", customer_data.columns)
        feature2 = st.sidebar.selectbox("Select Feature 2:", customer_data.columns)

        # Extract features for clustering
        X = customer_data[[feature1, feature2]].values

        # Predict DBSCAN cluster membership
        dbscan_model = DBSCAN().fit_predict(X)

        # Plot DBSCAN Clusters
        st.subheader("DBSCAN Clusters")
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0], X[:, 1], c=dbscan_model, cmap='viridis')
        plt.title('DBSCAN Clusters')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        st.pyplot(plt)
