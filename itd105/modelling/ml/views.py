from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as auth_login, logout
from .models import Mammo_Mass, Insurance
import json
import joblib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

def login(request):
    return render(request, 'signin.html')

def register(request):
    return render(request, 'signup.html')

def signup(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        User.objects.create_user(username=email, email=email, password=password)
        # Redirect to a success page or any other page
        return render(request, "signin.html")

    return render(request, 'signup.html')

def signin(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Authenticate the user
        user = authenticate(request, username=email, password=password)

        if user is not None:
            # Log in the user
            auth_login(request, user)
            
            # Redirect to a success page or any other page
            return render(request, 'landing.html')
        else:
            # Handle invalid login (display an error message or redirect back to the login page)
            return render(request, 'signin.html', {'error': 'Invalid login credentials'})

    return render(request, 'signin.html')

def signout(request):
    logout(request)
    return redirect('/')

@login_required
def eda(request):
    mammo_mass = Mammo_Mass.objects.all()
    mammo_mass_data = [{'BI_RADS': obj.BI_RADS, 'Age': obj.Age, 'Shape': obj.Shape, 'Margin': obj.Margin, 'Density': obj.Density, 'Severity': obj.Severity,} for obj in mammo_mass]
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(mammo_mass_data)
    df['Severity_Label'] = df['Severity'].map({0: 'Benign', 1: 'Malignant'})
    df_head_json = df.to_json(orient='split')

   # Get the frequency values
    severity_counts = df['Severity_Label'].value_counts().to_dict()
    age_counts = df['Age'].value_counts().sort_index().to_dict()

   # Prepare data for the pie chart
    labels_pie = list(severity_counts.keys())
    values_pie = list(severity_counts.values())

    # Create a pie chart using Plotly Express
    fig_pie = px.pie(names=labels_pie, values=values_pie, title='Severity Distribution')

    # Save the HTML representation of the pie chart
    pie_chart_html = fig_pie.to_html(full_html=False)

    # Prepare data for the line chart
    labels_line = list(age_counts.keys())
    values_line = list(age_counts.values())
    
   # Create a line chart using Plotly Express with customization
    fig_line = px.line(x=list(labels_line), y=list(values_line), title='Age Distribution',
                       labels={'x': 'Age Group', 'y': 'Count'},
                       line_shape='linear',  
                       render_mode='lines',  
                       template='plotly_dark')  
    
    # Add markers to the line chart
    fig_line.update_traces(mode='markers+lines', marker=dict(size=8, symbol='circle', line=dict(color='black', width=2)))

    # Save the HTML representation of the line chart
    line_chart_html = fig_line.to_html(full_html=False)


    return render(request, 'eda.html', {"df_head_json": json.loads(df_head_json),
                                         'pie_chart_html': pie_chart_html,
                                         'line_chart_html': line_chart_html})

@login_required
def class_add_view(request):
    return render (request, "class_add.html")

@login_required
def add_class(request):
    if request.method == 'POST':
        # Extract form data from the request
        bi_rads = int(request.POST.get('bi_rads'))
        age = int(request.POST.get('age'))
        shape = int(request.POST.get('shape'))
        margin = int(request.POST.get('margin'))
        density = int(request.POST.get('density'))
        severity = int(request.POST.get('severity'))

        # Create and save a new instance of YourModelName
        Mammo_Mass.objects.create(
            BI_RADS=bi_rads,
            Age=age,
            Shape=shape,
            Margin=margin,
            Density=density,
            Severity=severity
        )

        message = "Data Successfully Added"
        return render(request, 'class_add.html', {'message': message})

    return render(request, 'class_add.html')

@login_required
def train_class(request):
    return render(request, 'train.html')

@login_required
def model_train_class(request):
    if request.method == 'POST':
        mammo_mass = Mammo_Mass.objects.all()
        mammo_mass_data = [{'BI_RADS': obj.BI_RADS, 'Age': obj.Age, 'Shape': obj.Shape, 'Margin': obj.Margin, 'Density': obj.Density, 'Severity': obj.Severity,} for obj in mammo_mass]
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(mammo_mass_data)

        # Storing the data to the array
        array = df.values
        X = array[:, 0:5]
        Y = array[:, 5]

        # Set the test size and random seed for reproducibility
        test_size = 0.20
        seed = 5  # You can change this value

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # Initialize and train the Decision Tree Classifier with hyperparameters
        max_depth = int(request.POST.get('max_depth'))
        min_samples_split = int(request.POST.get('min_samples_split'))
        min_samples_leaf = int(request.POST.get('min_samples_leaf'))

        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed
        )

        # Train the model
        model.fit(X_train, Y_train)

        # Make predictions on the test set
        Y_pred = model.predict(X_test)

        # Specify the model file path
        model_directory = 'C:\\Users\\User\\Documents\\GitHub\\itd105-ni-ejb\\itd105\\modelling'
        os.makedirs(model_directory, exist_ok=True)  # Create the directory if it doesn't exist
        model_filename = os.path.join(model_directory, 'model_class.joblib')

        # Delete the old model file if it exists
        if os.path.exists(model_filename):
            os.remove(model_filename)

        # Save the trained model
        joblib.dump((model), model_filename)

        # Calculate accuracy
        accuracy = accuracy_score(Y_test, Y_pred)

        # Print or use the accuracy as needed
        print(f'Accuracy: {accuracy}')

        return render(request, 'train.html', {'accuracy':accuracy})

@login_required
def predict_class(request):
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'BI_RADS': float(request.POST.get('bi_rads')),
            'age': float(request.POST.get('age')),
            'Shape': int(request.POST.get('shape')),
            'Margin': float(request.POST.get('margin')),
            'Density': float(request.POST.get('density'))
        }

        # Create a DataFrame from the user input
        input_df = pd.DataFrame([user_input])

        # Define the path to the trained model
        model_directory = 'C:\\Users\\User\\Documents\\GitHub\\itd105-ni-ejb\\itd105\\modelling'
        model_filename = os.path.join(model_directory, 'model_class.joblib')

        # Load the trained model and label encoder
        model = joblib.load(model_filename)


        # Make predictions using the loaded model
        input_features = input_df.values
        prediction = model.predict(input_features)[0]

        # Map the prediction to a label
        predicted_label = 'Malignant' if prediction == 1 else 'Benign'

        # Pass the predicted label to the template
        print(predicted_label)
        return render(request, 'train.html', {'prediction': predicted_label})

    # Render the form for user input
    return render(request, 'train.html')

@login_required
def eda_reg(request):
    insurance = Insurance.objects.all()
    insurance_data = [{'age': obj.age, 'bmi': obj.bmi, 'children': obj.children, 'sex': obj.sex, 'smoker': obj.smoker, 'region': obj.region, 'charges': obj.charges,} for obj in insurance]
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(insurance_data)
    regions = df['region'].map({0: 'Southeast', 1: 'Southwest', 2: 'Northeast', 3: 'Northwest'})
    df_head_json = df.to_json(orient='split')
    smokers = df['smoker'].map({0: 'Yes', 1: 'No'})

    # Get the frequency values
    region_counts = regions.value_counts().to_dict()
    charges_counts = df['charges'].value_counts().to_dict()
    smoker_counts = smokers.value_counts().to_dict()
    # Prepare data for the pie chart
    labels_pie = list(region_counts.keys())
    values_pie = list(region_counts.values())
    # Create a pie chart using Plotly Express
    fig_pie = px.pie(names=labels_pie, values=values_pie, title='Regional Distribution')

    # Save the HTML representation of the pie chart
    pie_chart_html = fig_pie.to_html(full_html=False)

    labels_bar = list(smoker_counts.keys())
    values_bar = list(smoker_counts.values())

    # Create a bar chart
    fig_bar = go.Figure(go.Bar(
        x=labels_bar,
        y=values_bar,
        marker_color='rgb(26, 118, 255)'  # Customize the bar color
    ))

    # Update layout for better appearance
    fig_bar.update_layout(
        title='Smoker Counts',
        xaxis_title='Smoker Status',
        yaxis_title='Count',
        template='plotly_dark'  # Choose a template from: https://plotly.com/python/templates/
    )

    bar_chart_html = fig_bar.to_html(full_html=False)

    # Create a histogram
    fig_histogram = px.histogram(
        x=list(charges_counts.keys()),
        title='Age Distribution',
        labels={'x': 'Charges', 'y': 'Frequency'},
        template='plotly_dark',
        nbins=len(charges_counts)  # Set the number of bins based on the number of unique charges
    )

    # Save the HTML representation of the histogram
    histogram_html = fig_histogram.to_html(full_html=False)
    return render(request, 'eda_reg.html', {"df_head_json": json.loads(df_head_json),
                                            'pie_chart_html': pie_chart_html,
                                            'bar_chart_html': bar_chart_html,
                                            'histogram_html': histogram_html})

@login_required
def reg_add_view(request):
    return render (request, 'reg_add.html')

@login_required
def add_reg(request):
    if request.method == 'POST':
        # Extract form data from the request
        age = int(request.POST.get('age'))
        bmi = float(request.POST.get('bmi'))
        children = int(request.POST.get('children'))
        sex = int(request.POST.get('sex'))
        smoker = int(request.POST.get('smoker'))
        region = int(request.POST.get('region'))
        charges = float(request.POST.get('charges'))

        # Create and save a new instance of YourModelName
        Insurance.objects.create(
            age=age,
            bmi=bmi,
            children=children,
            sex=sex,
            smoker=smoker,
            region=region,
            charges=charges
        )
        message = "Data Successfully Added"
        return render(request, 'reg_add.html', {'message': message})

    return render(request, 'reg_add.html')

@login_required
def train_reg(request):
    return render(request, 'train_reg.html')

@login_required
def model_train_reg(request):
    if request.method == 'POST':
        insurance = Insurance.objects.all()
        insurance_data = [{'age': obj.age, 'bmi': obj.bmi, 'children': obj.children, 'sex': obj.sex, 'smoker': obj.smoker, 'region': obj.region, 'charges': obj.charges,} for obj in insurance]
        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(insurance_data)

        # Storing the data to the array
        array = df.values
        X = array[:, 0:6]
        Y = array[:, 6]

        # Set the test size and random seed for reproducibility
        test_size = 0.20
        seed = 5  # You can change this value

        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # Initialize and train the Decision Tree Classifier with hyperparameters
        n_estimators = int(request.POST.get('n_estimators'))
        min_samples_split = int(request.POST.get('min_samples_split'))
        min_samples_leaf = int(request.POST.get('min_samples_leaf'))

       # Train the data on a Random Forest Regressor with specified hyperparameters
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,       # Hyperparameter: The number of trees in the forest. You can adjust this for ensemble size.
            max_depth=None,        # Hyperparameter: The maximum depth of each tree. Adjust to control tree depth.
            min_samples_split=min_samples_split,    # Hyperparameter: The minimum number of samples required to split an internal node. Adjust to control node splitting.
            min_samples_leaf=min_samples_leaf,     # Hyperparameter: The minimum number of samples required in a leaf node. Adjust to control leaf size.
            random_state=seed        # Hyperparameter: Set a random seed for reproducibility.
        )

        # Train the model
        rf_model.fit(X_train, Y_train)

        # Make predictions on the test set
        Y_pred = rf_model.predict(X_test)

        # Specify the model file path
        model_directory = 'C:\\Users\\User\\Documents\\GitHub\\itd105-ni-ejb\\itd105\\modelling'
        os.makedirs(model_directory, exist_ok=True)  # Create the directory if it doesn't exist
        model_filename = os.path.join(model_directory, 'model_reg.joblib')

        # Delete the old model file if it exists
        if os.path.exists(model_filename):
            os.remove(model_filename)

        # Save the trained model
        joblib.dump((rf_model), model_filename)

        # Calculate the mean absolute error (MAE) for the predictions
        mae = mean_absolute_error(Y_test, Y_pred)
        print("MAE: %.3f" % mae)

        return render(request, 'train_reg.html', {'mae':mae})
    
@login_required
def predict_reg(request):
    if request.method == 'POST':
        # Get user input from the form
        user_input = {
            'age': int(request.POST.get('age')),
            'bmi': float(request.POST.get('bmi')),
            'children': int(request.POST.get('children')),
            'sex': int(request.POST.get('sex')),
            'smoker': int(request.POST.get('smoker')),
            'region': int(request.POST.get('region'))
        }

        # Create a DataFrame from the user input
        input_df = pd.DataFrame([user_input])

        # Define the path to the trained model
        model_directory = 'C:\\Users\\User\\Documents\\GitHub\\itd105-ni-ejb\\itd105\\modelling'
        model_filename = os.path.join(model_directory, 'model_reg.joblib')

        # Load the trained model and label encoder
        model = joblib.load(model_filename)


        # Make predictions using the loaded model
        input_features = input_df.values
        prediction = model.predict(input_features)[0]

        return render(request, 'train_reg.html', {'prediction': prediction})

    # Render the form for user input
    return render(request, 'train.html')