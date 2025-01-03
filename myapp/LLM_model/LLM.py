import replicate


# Initialize the Replicate client
def suggestions(age, gender, ap_hi, ap_lo, active, smoke, cholesterol, glucose, model_prediction):
    client = replicate.Client(
        api_token="r8_8vMPpCw9hEY5nZH3F5EFWHI3UH3pTzO3LtfV5"  # Replace with your actual API key
    )

    gender = 'male' if gender == 'M' else 'female'
    cholesterol_lvl = ""
    glucose_lvl = ""

    if cholesterol == 1:
        cholesterol_lvl = 'normal'
    elif cholesterol == 2:
        cholesterol_lvl = 'borderline high'
    elif cholesterol == 3:
        cholesterol_lvl = 'high'

    if glucose == 1:
        glucose_lvl = 'normal'
    elif glucose == 2:
        glucose_lvl = 'borderline high'
    elif glucose == 3:
        glucose_lvl = 'high'

    # Patient features
    features = {
        "age": age,
        "gender": gender,
        "ap_hi": ap_hi,  # Systolic blood pressure
        "ap_lo": ap_lo,   # Diastolic blood pressure
        "active": active,  # Physically active
        "smoke": smoke,    # Smoker
        "cholesterol_level": cholesterol_lvl,      # History of high cholesterol
        "glucose_level": glucose_lvl,    # Sugar level
        "model_pred": model_prediction
    }

    # Generate the prompt for the LLM
    prompt = f"""
    Patient Info:
    - Age: {features['age']}
    - Gender: {features['gender']}
    - BP: {features['ap_hi']}/{features['ap_lo']}
    - Active: {'Yes' if features['active'] else 'No'}, Smoker: {'Yes' if features['smoke'] else 'No'}
    - Conditions: Cholesterol Level: {cholesterol_lvl}, Sugar Level: {glucose_lvl}
    - Model Prediction for Cardiovascular Disease: {1 if features['model_pred'] else 0}
    
    Predict potential diseases based on the provided data and give tailored advice to improve the patient's health.
    """

    # Try to send the request to the model and get a response
    try:
        # Run the model prediction
        response = client.run(
            "meta/llama-2-70b-chat",  # Model identifier
            input={
                "top_k": 0,
                "top_p": 0.9,
                "prompt": prompt,
                "max_tokens": 512,
                "temperature": 0.3,
                "system_prompt": ("Analyze the patient's health data below to provide concise, actionable advice in under 200 words."
    
                "Start with an introductory sentence summarising the patient's health risks and overall status."
                "Provide 4–5 practical, tailored recommendations formatted as bullet points (each under 20 words)."
                "Predict potential health risks or diseases based on patient data with short explanations (2–3 sentences)."
                "End with a disclaimer urging the patient to consult a healthcare professional for final guidance."
                "Keep the tone professional, not overly formal."

                ),
                "length_penalty": 1,
                "max_new_tokens": 500,
                "presence_penalty": 0,
                "log_performance_metrics": False
            }
        )

        # Check if the response is a list and join the fragments
        if isinstance(response, list):
            response = ''.join(response)

        # Print the response as a formatted string
        print("\nPredicted Output:\n")
        print(response)
        response = '<h2 class="text-center">Suggestions</h2>' + response

        return response

    except Exception as e:
        # Handle any errors that occur during the API call
        print(f"Error occurred: {e}")
