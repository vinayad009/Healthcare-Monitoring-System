from math import sqrt

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import csv
from django.http import JsonResponse
import pickle
import numpy as np
from lightgbm import LGBMClassifier
import random
import os
from django.conf import settings
import pandas as pd
import xgboost as xgb
from .models import UserDetail
from django.views.decorators.csrf import csrf_exempt
from sklearn.preprocessing import StandardScaler
from .LLM_model.LLM import suggestions


@login_required
@csrf_exempt  # Allow CSRF for API-like functionality (ensure CSRF middleware is handled correctly)
def add_health_info(request):
    if request.method == "POST":
        user = request.user

        # Extract form data
        chol = int(request.POST.get('cholesterol'))
        glucose = int(request.POST.get('glucose'))
        smoke = int(request.POST.get('smoke'))
        alcohol = int(request.POST.get('alcohol'))
        age = int()
        bmi = float()
        high_bp = int()
        low_bp = int()
        body_temp = float()
        heart_rate = int()

        csv_file_path = 'data/locations.csv'  # Path to your CSV file

        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if row['name'] == request.user.first_name:
                    age1 = row['age']
                    bmi = row['bmi']
                    high_bp1 = row['blood_pressure_top']
                    low_bp1 = row['blood_pressure_bottom']
                    body_temp = row['body_temperature']
                    heart_rate = row['heart_rate']

        # Prepare data for prediction
        gender_numeric = 2 if user.userdetail.gender == 'M' else 1
        active = 0
        if int(heart_rate) > 120 and float(body_temp) > 36.2:
            active = 1
        else:
            active = 0

        weight = user.userdetail.weight
        age = user.userdetail.age
        high_bp = high_bp1
        low_bp = low_bp1
        height = int(sqrt(float(weight)/float(bmi)) * 100)

        age = int(age)
        gender_numeric = int(gender_numeric)
        height = int(height)
        weight = int(weight)
        high_bp = int(high_bp)
        low_bp = int(low_bp)
        chol = int(chol)
        glucose = int(glucose)
        smoke = int(smoke)
        alcohol = int(alcohol)
        active = int(active)

        try:
            # Load the LightGBM model
            model_path = os.path.join(settings.BASE_DIR, "lightgbm_model.pkl")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)

            # Prepare input features for prediction
            input_features = np.array([
                age, gender_numeric, height, weight, high_bp, low_bp, chol,
                glucose, smoke, alcohol, active
            ]).reshape(1, -1)

            # Predict using the loaded model
            prediction = model.predict(input_features)
            binary_prediction = 1 if prediction > 0.5 else 0
            prediction_message = binary_prediction

            print("Binary prediction: ", prediction_message)

            status = 0

            if prediction_message:
                status = 1

            summary = suggestions(
                age=age,
                gender=user.userdetail.gender,
                ap_hi=high_bp,
                ap_lo=low_bp,
                active=active,
                smoke=smoke,
                cholesterol=chol,
                glucose=glucose,
                model_prediction=status
            )

            print(summary)

            return JsonResponse({
                'status': 'success',
                'message': status,
                'summary': summary
            })
        except Exception as e:
            print(e)
            return JsonResponse({
                'status': 'error',
                'message': f'Prediction error: {str(e)}',
                'summary': 'None'
            }, status=500)

    else:
        # Render the form for GET requests
        return render(request, "home.html")


@login_required
@csrf_exempt  # Allow CSRF for API-like functionality (ensure CSRF middleware is handled correctly)
def add_full_health_info(request):
    if request.method == "POST":
        user = request.user
        chol = int(request.POST.get('cholesterol'))
        glucose = int(request.POST.get('glucose'))
        smoke = int(request.POST.get('smoke'))
        alcohol = int(request.POST.get('alcohol'))
        height = int(request.POST.get('height'))
        weight = int(request.POST.get('weight'))
        high_bp = int(request.POST.get('high_bp'))
        low_bp = int(request.POST.get('low_bp'))
        body_temp = float(request.POST.get('body_temp'))
        heart_rate = int(request.POST.get('heart_rate'))
        age = user.userdetail.age

        gender = user.userdetail.gender
        gender_numeric = 2 if gender == 'M' else 1

        active = int(0)
        if int(heart_rate) > 120 and float(body_temp) > 36.2:
            active = 1
        else:
            active = 0

        age = int(age)
        gender_numeric = int(gender_numeric)
        height = int(height)
        weight = int(weight)
        high_bp = int(high_bp)
        low_bp = int(low_bp)
        chol = int(chol)
        glucose = int(glucose)
        smoke = int(smoke)
        alcohol = int(alcohol)
        active = int(active)

        try:
            # Load the LightGBM model
            model_path = os.path.join(settings.BASE_DIR, "lightgbm_model.pkl")
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)

            # Prepare input features for prediction
            input_features = np.array([
                age, gender_numeric, height, weight, high_bp, low_bp, chol,
                glucose, smoke, alcohol, active
            ]).reshape(1, -1)

            # Predict using the loaded model
            prediction = model.predict(input_features)
            binary_prediction = 1 if prediction > 0.5 else 0
            prediction_message = binary_prediction

            print("Binary prediction: ", prediction_message)

            status = 0

            if prediction_message:
                status = 1

            summary = suggestions(
                age=age,
                gender=gender,
                ap_hi=high_bp,
                ap_lo=low_bp,
                active=active,
                smoke=smoke,
                cholesterol=chol,
                glucose=glucose,
                model_prediction=status
            )

            print(summary)

            return JsonResponse({
                'status': 'success',
                'message': status,
                'summary': summary
            })

        except Exception as e:
            print(e)
            return JsonResponse({
                'status': 'error',
                'message': f'Prediction error: {str(e)}',
                'summary': 'None'
            }, status=500)

    else:
        # Render the form for GET requests
        return render(request, "home.html")


def get_user_data(request):
    username = request.GET.get("username")
    if not username:
        return JsonResponse({"error": "Username is required"}, status=400)

    # Assuming the data comes from the CSV or database
    csv_file_path = "data/locations.csv"
    userData = {"timestamps": [], "low_bp": [], "high_bp": [], "heart_rate": [], "body_temp": []}

    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["name"] == username:
                userData["timestamps"].append(row["timestamp"])
                userData["low_bp"].append(float(row["blood_pressure_bottom"]))
                userData["high_bp"].append(float(row["blood_pressure_top"]))
                userData["heart_rate"].append(float(row["heart_rate"]))
                userData["body_temp"].append(float(row["body_temperature"]))

    return JsonResponse(userData)


@login_required
def home(request):
    user = request.user
    if request.user.is_superuser:
        # xgb_prediction = None
        locations = []  # List to store extracted location data
        csv_file_path = 'data/locations.csv'  # Path to your CSV file
        username = set()
        alert_counts = {"warning": 0, "emergency": 0}
        disease_counts = {  # Initialize disease counts
            "hypertension": 0,
            "hyperglycemia": 0,
            "fever": 0,
            "hyperthermia": 0,
            "hypothermia": 0,
            "tachycardia": 0
        }
        user_data = []
        # gender = None
        # age = None
        # bmi = None
        # high_bp = None
        # low_bp = None
        # body_temp = None
        # heart_rate = None

        xgb_model_path = os.path.join(settings.BASE_DIR, "trained_xgb_model.pkl")
        with open(xgb_model_path, 'rb') as model_file:
            xgb_model = pickle.load(model_file)

        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if i == 1:
                    name = row['name']
                    gender = row['gender']
                    age = row['age']
                    bmi = row['bmi']
                    high_bp = row['blood_pressure_top']
                    low_bp = row['blood_pressure_bottom']
                    body_temp = row['body_temperature']
                    heart_rate = row['heart_rate']

                try:
                    # Collect only the relevant data
                    username.add(row['name'])
                    location_data = {
                        'name': row['name'],
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude']),
                        'alert': row.get('alert', '')  # Add alert column if present
                    }
                    if row['alert']:
                        if not any(loc['name'] == row['name'] for loc in user_data):
                            user_data.append({'name': row['name'],
                                              'alert': row['alert'],
                                              'age': row['age'],
                                              'gender': row['gender'],
                                              'heart_rate': row['heart_rate'],
                                              'high_bp': row['blood_pressure_top'],
                                              "low_bp": row['blood_pressure_bottom'],
                                              "body_temp": row['body_temperature']
                                              })

                    locations.append(location_data)
                    alert_type = location_data['alert'].split('!')[0].strip().lower()
                    if alert_type in alert_counts:
                        alert_counts[alert_type] += 1

                    alert_words = location_data['alert'].split()
                    if len(alert_words) > 1:
                        disease = alert_words[1].lower()  # Second word is expected to be the disease
                        # Increment disease count if it matches any of the known diseases
                        if disease in disease_counts:
                            disease_counts[disease] += 1

                except (ValueError, KeyError) as e:
                    print(f"Error processing row: {row}, Error: {e}")

        # Debugging output to validate JSON structure
        gender_numeric = 2 if gender == 'Male' else 1
        active = 0
        if int(heart_rate) > 120 and float(body_temp) > 36.2:
            active = 1
        else:
            active = 0

        weight = UserDetail.objects.values_list('weight').get(user__first_name=name)[0]
        high_bp = high_bp
        low_bp = low_bp
        height = int(sqrt(float(weight) / float(bmi)) * 100)

        xgb_input = np.array([[
            int(age),
            int(gender_numeric),
            int(height),
            int(weight),
            int(high_bp),
            int(low_bp),
            active
        ]])

        xgb_prediction = xgb_model.predict(xgb_input)[0]

        print(xgb_prediction)
        length = len(locations)
        loca = locations[:1000]
        number_users = len(username)
        print(f"Alert counts: {alert_counts}")
        print(f"Disease counts: {disease_counts}")
        print("Final locations data:", locations[0])
        print("lenght of dataset:", length)
        print("Length:", len(username))
        print("length of user_alert:", len(user_data))
        warning = alert_counts['warning']
        emergency = alert_counts['emergency']
        total_alerts = warning + emergency
        return render(request, 'home.html',
                      {'user': request.user,
                       'locations': loca,
                       'length': length,
                       'number_users': number_users,
                       'warning': warning,
                       'emergency': emergency,
                       'total_alerts': total_alerts,
                       'disease_counts': disease_counts,
                       'user_data': user_data,
                       'xgb_prediction': int(xgb_prediction)})

    else:
        # xgb_prediction = None
        locations = []  # List to store extracted location data
        csv_file_path = 'data/locations.csv'  # Path to your CSV file
        first_name = request.user.first_name
        alert_counts = {"warning": 0, "emergency": 0}
        disease_counts = {  # Initialize disease counts
            "hypertension": 0,
            "hyperglycemia": 0,
            "fever": 0,
            "hyperthermia": 0,
            "hypothermia": 0,
            "tachycardia": 0
        }
        user_data = []

        xgb_model_path = os.path.join(settings.BASE_DIR, "trained_xgb_model.pkl")
        with open(xgb_model_path, 'rb') as model_file:
            xgb_model = pickle.load(model_file)

        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                if row['name'] == first_name:
                    age = row['age']
                    bmi = row['bmi']
                    high_bp = row['blood_pressure_top']
                    low_bp = row['blood_pressure_bottom']
                    body_temp = row['body_temperature']
                    heart_rate = row['heart_rate']

                    try:
                        # Collect only the relevant data
                        location_data = {
                            'name': row['name'],
                            'latitude': float(row['latitude']),
                            'longitude': float(row['longitude']),
                            'alert': row.get('alert', '')  # Add alert column if present
                        }
                        if row['alert']:
                            user_data.append({'alert': row['alert'],
                                              'heart_rate': row['heart_rate'],
                                              'bp_hi': row['blood_pressure_top'],
                                              'bp_lo': row['blood_pressure_bottom'],
                                              'body_temp': row['body_temperature'],
                                              'age': row['age'],
                                              'gender': row['gender']
                                              })

                        locations.append(location_data)
                        alert_type = location_data['alert'].split('!')[0].strip().lower()
                        if alert_type in alert_counts:
                            alert_counts[alert_type] += 1

                        alert_words = location_data['alert'].split()
                        if len(alert_words) > 1:
                            disease = alert_words[1].lower()  # Second word is expected to be the disease
                            # Increment disease count if it matches any of the known diseases
                            if disease in disease_counts:
                                disease_counts[disease] += 1

                    except (ValueError, KeyError) as e:
                        print(f"Error processing row: {row}, Error: {e}")

        # Debugging output to validate JSON structure
        gender_numeric = 2 if user.userdetail.gender == 'M' else 1
        active = 0
        if int(heart_rate) > 120 and float(body_temp) > 36.2:
            active = 1
        else:
            active = 0

        weight = user.userdetail.weight
        user.userdetail.age = age
        high_bp = high_bp
        low_bp = low_bp
        height = int(sqrt(float(weight) / float(bmi)) * 100)

        xgb_input = np.array([[
            int(age),
            gender_numeric,
            height,
            int(weight),
            int(high_bp),
            int(low_bp),
            active
        ]])

        xgb_prediction = xgb_model.predict(xgb_input)[0]

        print(xgb_prediction)
        length = len(locations)
        number_users = len(user_data)
        print(f"Alert counts: {alert_counts}")
        print(f"Disease counts: {disease_counts}")
        print("Final locations data:", locations[0])

        print("lenght of dataset:", length)
        print("length of user_alert:", len(user_data))
        warning = alert_counts['warning']
        emergency = alert_counts['emergency']
        total_alerts = warning + emergency
        return render(request, 'home.html',
                      {'user': request.user,
                       'locations': locations,
                       'length': length,
                       'number_users': number_users,
                       'warning': warning,
                       'emergency': emergency,
                       'total_alerts': total_alerts,
                       'disease_counts': disease_counts,
                       'user_data': user_data,
                       'xgb_prediction': int(xgb_prediction)})


def login_view(request, organisation):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        super_user = False
        if organisation:
            super_user = True

        # Fetch the user by email
        try:
            user = User.objects.get(email=email, is_superuser=super_user)  # Get user based on email
        except User.DoesNotExist:
            user = None

        if user:
            # Authenticate using the username associated with the email
            user = authenticate(request, username=user.username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')  # Redirect to home page
            else:
                messages.error(request, 'Invalid email or password')
        else:
            messages.error(request, 'Invalid email or password')

    if organisation:
        return render(request, 'organisation_login.html')
    return render(request, 'login.html')


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        name = request.POST['name']
        gender = request.POST['gender']
        weight = request.POST['weight']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
        elif User.objects.filter(email=email).exists():
            messages.error(request, 'Email already exists')
        else:
            # Create a new user
            user = User.objects.create_user(
                first_name=name,
                username=email,  # Use email as the username
                email=email,
                password=password
            )

            # Update profile details
            profile_ = user.userdetail
            profile_.gender = 'M' if gender == 'Male' else 'F'
            profile_.weight = weight
            profile_.save()

            user.save()
            login(request, user)
            return redirect('home')

    return render(request, 'signup.html')


def logout_view(request):
    logout(request)
    return redirect('home')


def profile(request):
    if request.method == 'POST':
        user = request.user
        user.email = request.POST.get('email')

        # Update profile fields (ensure a Profile model exists)
        profile_ = UserDetail.objects.get(user=user)
        profile_.weight = request.POST.get('weight')
        profile_.age = request.POST.get('age')

        print(request.POST.get('age'))
        profile_.save(update_fields=['weight', 'age'])

        user.save(update_fields=['email'])
        return redirect('profile')  # Redirect back to the profile page
    return render(request, 'profile.html')
