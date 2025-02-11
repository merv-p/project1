import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import seaborn as sns


# Function to generate dummy data
def generate_dummy_data():
    np.random.seed(42)
    group = np.random.choice(["A", "B"], size=100)
    feature1 = np.random.normal(50, 10, 100)  # Normally distributed data
    feature2 = feature1 * 0.8 + np.random.normal(0, 5, 100)  # Correlated variable
    feature3 = np.random.normal(30, 8, 100)  # Independent variable

    df = pd.DataFrame({"Group": group, "Feature1": feature1, "Feature2": feature2, "Feature3": feature3})
    return df


# Load dummy data
df = generate_dummy_data()

# Streamlit UI
st.title("Basic Data Analysis with Streamlit")

# Sidebar for navigation
st.sidebar.header("Navigation")
option = st.sidebar.radio("Select Analysis:",
                          ["Dataset Overview", "T-Test", "Canonical Correlation Analysis", "Definitions"])

if option == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("### Sample Data:")
    st.write(df.head())
    st.write("### Summary Statistics:")
    st.write(df.describe())

    # Data visualization
    st.write("### Data Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df, bins=20, kde=True, ax=ax)
    st.pyplot(fig)

elif option == "T-Test":
    st.header("T-Test Analysis")
    st.write("A T-test compares the means of two independent groups.")

    # Separate data by group
    group_A = df[df["Group"] == "A"]["Feature1"]
    group_B = df[df["Group"] == "B"]["Feature1"]

    # Perform T-test
    t_stat, p_value = stats.ttest_ind(group_A, group_B)

    st.write(f"T-Statistic: {t_stat:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    if p_value < 0.05:
        st.write("### Conclusion: The difference is statistically significant.")
    else:
        st.write("### Conclusion: The difference is not statistically significant.")

    # Visualization
    st.write("### Boxplot of Groups")
    fig, ax = plt.subplots()
    sns.boxplot(x="Group", y="Feature1", data=df, ax=ax)
    st.pyplot(fig)

elif option == "Canonical Correlation Analysis":
    st.header("Canonical Correlation Analysis (CCA)")
    st.write("CCA finds relationships between two sets of variables.")

    # Selecting features
    X = df[["Feature1", "Feature2"]]
    Y = df[["Feature3"]]

    # Perform CCA
    cca = CCA(n_components=1)
    X_c, Y_c = cca.fit_transform(X, Y)
    correlation = np.corrcoef(X_c.T, Y_c.T)[0, 1]

    st.write(f"Canonical Correlation: {correlation:.4f}")

    # Visualization
    st.write("### Canonical Variables Relationship")
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_c[:, 0], y=Y_c[:, 0], ax=ax)
    ax.set_xlabel("Canonical Variable X")
    ax.set_ylabel("Canonical Variable Y")
    st.pyplot(fig)

elif option == "Definitions":
    st.header("Definitions of Terms")
    st.write("**T-Test:** A statistical test used to compare the means of two groups.")
    st.write(
        "**Canonical Correlation Analysis (CCA):** A method to explore relationships between two sets of variables.")
    st.write(
        "**P-Value:** Probability of obtaining results at least as extreme as observed, assuming the null hypothesis is true.")
    st.write("**Histogram:** A graphical representation of data distribution.")
    st.write("**Boxplot:** A graphical method to show data spread and detect outliers.")
