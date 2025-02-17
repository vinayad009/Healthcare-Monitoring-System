import requests
import json
import re

def send_whatsapp_message(phone_number, name, binary_prediction, summary):
    if binary_prediction:
        prediction = "Positive"
    else:
        prediction = "Negative"

    single_line_summary = re.sub(r"[\n\t]+", " ", summary).strip()
    single_line_summary = re.sub(r"<h2.*?>.*?</h2>", "", single_line_summary , flags=re.DOTALL).strip()
    # url = "https://graph.facebook.com/v21.0/534516213081946/messages"

    # payload = json.dumps({
    #     "messaging_product": "whatsapp",
    #     "to": f"{phone_number}",
    #     "type": "template",
    #     "template": {
    #         "name": "hello_world",
    #         "language": {
    #             "code": "en_US"
    #         },
    #         "components": [
    #                 {
    #                     "type": "body",
    #                     "parameters": [
    #                         {"type": "text", "text": f"{name}"},
    #                         {"type": "text", "text": f"{prediction}"},
    #                         {"type": "text", "text": f"{summary}"}
    #                     ]
    #                 }
    #         ]
    #     }
    # })

    # payload = f"""{{
    #     "messaging_product": "whatsapp",
    #     "to": "918884552919",
    #     "type": "template",
    #     "template": {{
    #         "name": "health_alert",
    #         "language": {{
    #             "code": "en"
    #         }},
    #         "components": [
    #             {{
    #                 "type": "body",
    #                 "parameters": [
    #                     {{ "type": "text", "text": "{name}" }},
    #                     {{ "type": "text", "text": "{prediction}" }},
    #                     {{ "type": "text", "text": "{summary}" }}
    #                 ]
    #             }}
    #         ]
    #     }}
    # }}"""
    #
    # headers = {
    #     'Authorization': 'Bearer EAAMzVa9yfkoBO4jdOsT3k7IZBZBx4sjpAIHhIpvwdGh2YpjRoxezvzsijKARO5lOysrv3StOw4BAfuh3tI7HE2J6ZBYyX0L7kEb4ScMWJRCCc2vKsNkFrZAQn506ZAluRg0UWuGdpOeBndL6bv3oRf7G3P8IypCbdDQOuONL858bxYM59Wpu4lnMNpeWYKV3rPHzQ6qxTVpHnE2NJ9O9WGtbtpfv7ugZDZD',
    #     'Content-Type': 'application/json'
    # }

    url = "https://graph.facebook.com/v21.0/534516213081946/messages"

    payload = json.dumps({
        "messaging_product": "whatsapp",
        "to": "918884552919",
        "type": "template",
        "template": {
            "name": "health_alert",
            "language": {
                "code": "en"
            },
            "components": [
                {
                    "type": "body",
                    "parameters": [
                        {
                            "type": "text",
                            "text": name
                        },
                        {
                            "type": "text",
                            "text": prediction
                        },
                        {
                            "type": "text",
                            "text": single_line_summary[:500]
                        }
                    ]
                }
            ]
        }
    })
    headers = {
        'Authorization': 'Bearer EAAMzVa9yfkoBO2Od1h8Y9VZCi8fhmHXr4ZBSaPUcmgGzYojbmnZAzbCvOZAlWWGtrxzwkBI7QdreQBTHyDAl59RFywAdYEjNPXDF7FeWJGr07ZAdlZAjyXTRHdxhTLVEOxtHH5V15AIlY3TFT1Own57eZAxXvhLNUmwZA7nJklpNs3wVmyEVAecJXgJVM7FXZBf00aQXL8t3rwyXNUpTAYDOEOXksWZAYZD',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        print("Response Status:", response.status_code)
        print("Response Body:", response.text)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending WhatsApp message: {e}")
        return None
