from django.shortcuts import render

# Create your views here.
# chatbot/views.py

from rest_framework.response import Response
from rest_framework.decorators import api_view
from .chatbot_logic import classify_intent_input, generate_response

@api_view(['POST'])
def chatbot_view(request):
    user_input = request.data.get('user_input', '')  # Assuming user input is sent in the 'user_input' field
    if user_input.lower() == "exit":
        response = {"chatbot_response": "Goodbye!"}
    else:
        intent = classify_intent_input(user_input)
        response_text = generate_response(user_input, intent,)
        response = {"chatbot_response": response_text}
    return Response(response)

