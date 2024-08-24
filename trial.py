import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_classif, f_regression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

# Set page config
st.set_page_config(page_title="Data analysis App", page_icon="📊", layout="wide")



# Web App Title
st.markdown('''
    <h1 style="color:#FFA500;">Smart Data Analysis and Predictive Modeling Application</h1>
    <p style="font-size:18px;">  Welcome to the smart world. </p>
    <hr style="border-top: 2px solid #F63366;">
''', unsafe_allow_html=True)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = ""
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}



# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a Section", ["Upload Dataset", "Data Overview", "Descriptive Statistics", "Data Cleaning", "Convert Data","EDA", "Normalize Data", "Feature Selection",  "Model Building", "Model Comparison", "Hyperparameter Tuning", "Export Results"])

def upload_data():
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(st.session_state.df.head())

def data_overview():
    st.header("Data Overview")
    if not st.session_state.df.empty:
        st.write("Shape of the dataset:", st.session_state.df.shape)
        st.write("Data Types:")
        st.write(st.session_state.df.dtypes)
        st.write("Missing Values:")
        st.write(st.session_state.df.isnull().sum())
    else:
        st.write("No dataset uploaded yet.")

def descriptive_statistics():
    st.header("Descriptive Statistics")
    if not st.session_state.df.empty:
        st.write(st.session_state.df.describe())
    else:
        st.write("No dataset uploaded yet.")

def handle_missing_values():
    st.header("Handle Missing Values")
    if not st.session_state.df.empty:
        missing_option = st.selectbox("Select an option", ["None", "Drop missing values", "Fill missing values"])
        if missing_option == "Drop missing values":
            st.session_state.df = st.session_state.df.dropna()
        elif missing_option == "Fill missing values":
            fill_value = st.text_input("Enter value to fill missing values", "0")
            st.session_state.df = st.session_state.df.fillna(fill_value)
        st.write("Dataset after handling missing values:")
        st.write(st.session_state.df.head())
    else:
        st.write("No dataset uploaded yet.")

def convert_categorical_to_numeric():
    st.header("Convert Non-Numeric Values to Numeric")
    if not st.session_state.df.empty:
        categorical_cols = st.session_state.df.select_dtypes(include=['object']).columns
        if categorical_cols.size > 0:
            st.write("Categorical columns detected:")
            st.write(categorical_cols.tolist())
            columns_to_convert = st.multiselect("Select columns to convert to numeric", categorical_cols.tolist())
            if columns_to_convert:
                le = LabelEncoder()
                df_converted = st.session_state.df.copy()
                for col in columns_to_convert:
                    df_converted[col] = le.fit_transform(df_converted[col])
                st.session_state.df = df_converted
                st.write("Dataset after conversion:")
                st.write(st.session_state.df.head())
            else:
                st.write("No columns selected for conversion.")
        else:
            st.write("No categorical columns to convert.")
    else:
        st.write("No dataset uploaded yet.")

def normalize_data():
    st.header("Normalize/Scale Data")
    if not st.session_state.df.empty:
        columns_to_scale = st.multiselect("Select columns to normalize/scale", st.session_state.df.columns)
        scaler_option = st.selectbox("Choose Scaling Technique", ["None", "StandardScaler", "MinMaxScaler"])
        if scaler_option != "None" and columns_to_scale:
            if scaler_option == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_option == "MinMaxScaler":
                scaler = MinMaxScaler()
            st.session_state.df[columns_to_scale] = scaler.fit_transform(st.session_state.df[columns_to_scale])
            st.write("Data after normalization/scaling:")
            st.write(st.session_state.df.head())
    else:
        st.write("No dataset uploaded yet.")

def feature_selection():
    st.header("Feature Selection")
    if not st.session_state.df.empty:
        target = st.selectbox("Select Target Variable", st.session_state.df.columns)
        st.session_state.target_variable = target
        
        task_type = st.selectbox("Select Task Type", ["Regression", "Classification"])
        
        if target and task_type:
            X = st.session_state.df.drop(columns=[target])
            y = st.session_state.df[target]
            
            if task_type == "Classification" and y.dtype == 'O':
                st.write("Please convert the target variable to numeric for classification.")
                return
            
            if task_type == "Classification":
                f_values, p_values = f_classif(X, y)
            elif task_type == "Regression":
                f_values, p_values = f_regression(X, y)
            
            feature_scores = pd.DataFrame({
                'Feature': X.columns,
                'F-Value': f_values,
                'P-Value': p_values
            })
            
            feature_scores = feature_scores.sort_values(by='F-Value', ascending=False)
            
            st.write("Top Features Based on ANOVA Test:")
            st.write(feature_scores)
            
            features = st.multiselect("Select Feature Variables", feature_scores['Feature'])
            st.session_state.selected_features = features
            
            if features:
                X_selected = st.session_state.df[features]
                st.write("Selected Features:")
                st.write(X_selected.head())
            else:
                st.write("No features selected.")
        else:
            st.write("Please provide target variable and task type.")
    else:
        st.write("No dataset uploaded yet.")

def exploratory_data_analysis():
    st.header("Exploratory Data Analysis (EDA)")
    if not st.session_state.df.empty:
        graph_type = st.selectbox("Select Graph Type", ["None", "Histogram", "Scatter Plot", "Correlation Matrix","Bar Plot","Line Plot","Count Plot"])
        
        if graph_type == "Histogram":
            column = st.selectbox("Select Column for Histogram", st.session_state.df.columns)
            if column:
                fig, ax = plt.subplots()
                ax.hist(st.session_state.df[column].dropna(), bins=30, edgecolor='k')
                ax.set_title(f'Histogram of {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                st.pyplot(fig)
        
        elif graph_type == "Scatter Plot":
            x_col = st.selectbox("Select X Column", st.session_state.df.columns)
            y_col = st.selectbox("Select Y Column", st.session_state.df.columns)
            if x_col and y_col:
                fig, ax = plt.subplots()
                ax.scatter(st.session_state.df[x_col], st.session_state.df[y_col])
                ax.set_title(f'Scatter Plot of {x_col} vs {y_col}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
        
        elif graph_type == "Correlation Matrix":
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = st.session_state.df.corr()
            sns.heatmap(corr, 
                        annot=True,
                        cmap='coolwarm',
                        fmt='.2f',
                        square=True,
                        linewidths=0.5,
                        cbar_kws={"shrink": .8},
                        ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
        elif graph_type == "Bar Plot":
            column = st.selectbox("Select Column for Bar Plot", st.session_state.df.columns)
            if column:
                fig, ax = plt.subplots()
                st.session_state.df[column].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f'Bar Plot of {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Count')
                st.pyplot(fig)
                
        elif graph_type == "Line Plot":
            x_col = st.selectbox("Select X Column for Line Plot", st.session_state.df.columns)
            y_col = st.selectbox("Select Y Column for Line Plot", st.session_state.df.columns)
            if x_col and y_col:
                fig, ax = plt.subplots()
                ax.plot(st.session_state.df[x_col], st.session_state.df[y_col])
                ax.set_title(f'Line Plot of {y_col} over {x_col}')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                st.pyplot(fig)
                
        elif graph_type == "Count Plot":
            column = st.selectbox("Select Column for Count Plot", st.session_state.df.columns)
            if column:
                fig, ax = plt.subplots()
                sns.countplot(data=st.session_state.df, x=column, ax=ax)
                ax.set_title(f'Count Plot of {column}')
                st.pyplot(fig)
    else:
        st.write("No dataset uploaded yet.")

def model_building():
    st.header("Model Building")
    if not st.session_state.df.empty:
        if st.session_state.target_variable and st.session_state.selected_features:
            # Display selected target and features for verification
            st.subheader("Selected Target and Features")
            st.write(f"Target Variable: {st.session_state.target_variable}")
            st.write(f"Selected Features: {', '.join(st.session_state.selected_features)}")
            
            confirm = st.radio("Is this correct?", ["Yes", "No"])
            if confirm == "No":
                st.write("Please go back to the Feature Selection step to adjust your choices.")
                return
        
        task_type = st.selectbox("Select Task Type for Model Building", ["None", "Regression", "Classification"])
        
        if task_type == "Regression" and st.session_state.target_variable:
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_variable]
            
            if len(X.columns) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model_type = st.selectbox("Select Regression Model", [
                    "None", 
                    "Linear Regression", 
                    "Polynomial Regression", 
                    "Support Vector Regression",
                    "Decision Tree Regression",
                    "Random Forest Regression",
                    "Gradient Boosting Regression",
                    "K-Nearest Neighbors Regression"
                ])
                
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Polynomial Regression":
                    degree = st.slider("Select Polynomial Degree", 2, 5, 2)
                    poly = PolynomialFeatures(degree=degree)
                    X_poly = poly.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
                    model = LinearRegression()
                elif model_type == "Support Vector Regression":
                    from sklearn.svm import SVR
                    kernel = st.selectbox("Select Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    model = SVR(kernel=kernel)
                elif model_type == "Decision Tree Regression":
                    from sklearn.tree import DecisionTreeRegressor
                    model = DecisionTreeRegressor()
                elif model_type == "Random Forest Regression":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor()
                elif model_type == "Gradient Boosting Regression":
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor()
                elif model_type == "K-Nearest Neighbors Regression":
                    from sklearn.neighbors import KNeighborsRegressor
                    n_neighbors = st.slider("Select Number of Neighbors", 1, 20, 5)
                    model = KNeighborsRegressor(n_neighbors=n_neighbors)
                else:
                    st.write("Select a regression model.")
                    return
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.session_state.model_results = {
                    'Mean Squared Error': mean_squared_error(y_test, y_pred),
                    'R2 Score': r2_score(y_test, y_pred),
                    'Mean Absolute Error': mean_absolute_error(y_test, y_pred)
                }
                
                st.write("Model Evaluation Metrics:")
                st.write(st.session_state.model_results)
                
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5)
                ax.set_title('Actual vs Predicted Values')
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                st.pyplot(fig)
                
        elif task_type == "Classification" and st.session_state.target_variable:
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_variable]
            
            # Check if target variable is continuous
            if pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 10:  # Arbitrary threshold for continuous target
                st.error("The selected target variable is continuous. Please select a regression task instead.")
                return
            
            if len(X.columns) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model_type = st.selectbox("Select Classification Model", [
                    "None", 
                    "Logistic Regression", 
                    "Decision Tree", 
                    "Random Forest", 
                    "SVM"
                ])
                
                if model_type == "Logistic Regression":
                    model = LogisticRegression()
                elif model_type == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif model_type == "Random Forest":
                    model = RandomForestClassifier()
                elif model_type == "SVM":
                    from sklearn.svm import SVC
                    model = SVC()
                else:
                    st.write("Select a classification model.")
                    return
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                st.session_state.model_results = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Classification Report': classification_report(y_test, y_pred),
                    'Confusion Matrix': confusion_matrix(y_test, y_pred)
                }
                
                st.write("Model Evaluation Metrics:")
                st.write(st.session_state.model_results['Accuracy'])
                st.text(st.session_state.model_results['Classification Report'])
                
                fig, ax = plt.subplots()
                sns.heatmap(st.session_state.model_results['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
        else:
            st.write("Select task type and ensure target variable is set.")
    else:
        st.write("No dataset uploaded yet.")


def model_comparison():
    st.header("Model Comparison")
    if not st.session_state.df.empty:
        task_type = st.selectbox("Select Task Type for Comparison", ["None", "Regression", "Classification"])
        
        if task_type == "Regression" and st.session_state.target_variable:
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            models = {
                "Linear Regression": LinearRegression(),
                "Polynomial Regression": Pipeline([
                    ('poly', PolynomialFeatures()),
                    ('linear', LinearRegression())
                ]),
                "Decision Tree Regression": DecisionTreeRegressor(),
                "Random Forest Regression": RandomForestRegressor(),
                "Gradient Boosting Regression": GradientBoostingRegressor(),
                "K-Nearest Neighbors Regression": KNeighborsRegressor()
            }
            
            results = {}
            for name, model in models.items():
                if name == "Polynomial Regression":
                    for degree in [2, 3, 4]:
                        model.set_params(poly__degree=degree)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results[f"{name} (Degree {degree})"] = {"MSE": mse, "R2": r2}
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results[name] = {"MSE": mse, "R2": r2}
            
            st.subheader("Model Comparison Results")
            for name, metrics in results.items():
                st.write(f"**{name}:** MSE: {metrics['MSE']:.2f}, R2: {metrics['R2']:.2f}")
        
        elif task_type == "Classification" and st.session_state.target_variable:
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC()
            }
            
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy
            
            st.subheader("Model Comparison Results")
            for name, accuracy in results.items():
                st.write(f"**{name}:** Accuracy: {accuracy:.2f}")

        else:
            st.write("Select task type and ensure target variable is set.")
    else:
        st.write("No dataset uploaded yet.")

def hyperparameter_tuning():
    st.header("Hyperparameter Tuning")
    if not st.session_state.df.empty:
        task_type = st.selectbox("Select Task Type for Hyperparameter Tuning", ["None", "Regression", "Classification"])
        
        if task_type == "Regression" and st.session_state.target_variable:
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model_type = st.selectbox("Select Regression Model for Hyperparameter Tuning", [
                "None",
                "Linear Regression",
                "Decision Tree Regression",
                "Random Forest Regression",
                "Gradient Boosting Regression"
            ])
            
            param_grids = {
                "Decision Tree Regression": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest Regression": {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 10, 20, 30]
                },
                "Gradient Boosting Regression": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7]
                }
            }
            
            if model_type in param_grids:
                model = {
                    "Decision Tree Regression": DecisionTreeRegressor(),
                    "Random Forest Regression": RandomForestRegressor(),
                    "Gradient Boosting Regression": GradientBoostingRegressor()
                }[model_type]
                
                grid_search = GridSearchCV(model, param_grids[model_type], cv=5, scoring='neg_mean_squared_error')
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                
                st.write(f"Best Parameters for {model_type}: {grid_search.best_params_}")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
                st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

        elif task_type == "Classification" and st.session_state.target_variable:
            X = st.session_state.df[st.session_state.selected_features]
            y = st.session_state.df[st.session_state.target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model_type = st.selectbox("Select Classification Model for Hyperparameter Tuning", [
                "None",
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "SVM"
            ])
            
            param_grids = {
                "Logistic Regression": {
                    'C': [0.1, 1, 10]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [None, 10, 20, 30]
                },
                "SVM": {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf']
                }
            }
            
            if model_type in param_grids:
                model = {
                    "Logistic Regression": LogisticRegression(),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "SVM": SVC()
                }[model_type]
                
                grid_search = GridSearchCV(model, param_grids[model_type], cv=5, scoring='accuracy')
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                
                st.write(f"Best Parameters for {model_type}: {grid_search.best_params_}")
                st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
                st.text(classification_report(y_test, y_pred))
                
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

        else:
            st.write("Select task type and ensure target variable is set.")
    else:
        st.write("No dataset uploaded yet.")


def export_results():
    st.header("Export Results")
    if 'df' in st.session_state and not st.session_state.df.empty:
        st.write("Processed Dataset:")
        st.write(st.session_state.df.head())

        # Download the processed dataset
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="Download Processed Dataset",
            data=csv,
            file_name="processed_dataset.csv",
            mime="text/csv"
        )
        
        # Download the analysis results
        if 'analysis_results' in st.session_state and st.session_state.analysis_results:
            analysis_results = st.session_state.analysis_results
            st.download_button(
                label="Download Analysis Results",
                data=analysis_results,
                file_name="analysis_results.txt",
                mime="text/plain"
            )
        else:
            st.write("No analysis results available to download.")

# Conditional page rendering based on selected sidebar option
if options == "Upload Dataset":
    upload_data()
elif options == "Data Overview":
    data_overview()
elif options == "Descriptive Statistics":
    descriptive_statistics()
elif options == "Data Cleaning":
    handle_missing_values()
elif options == "Convert Data":
    convert_categorical_to_numeric()
elif options == "EDA":
    exploratory_data_analysis()
elif options == "Normalize Data":
    normalize_data()
elif options == "Feature Selection":
    feature_selection()
elif options == "Model Building":
    model_building()
elif options == "Model Comparison":
    model_comparison()
elif options == "Hyperparameter Tuning":
    hyperparameter_tuning()
elif options == "Export Results":
    export_results()

# Function to generate a text report
def generate_report():
    report = "Analysis Report\n\n"
    if st.session_state.analysis_results:
        report += "Data Overview:\n" + str(st.session_state.analysis_results) + "\n\n"
    if st.session_state.model_results:
        report += "Model Results:\n" + str(st.session_state.model_results) + "\n\n"
    return report

# Store the report in session state
if st.sidebar.button("Generate Report"):
    st.session_state.analysis_results = generate_report()
