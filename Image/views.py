
from django.shortcuts import render
from io import BytesIO
import base64
import os
import uuid
import torch
import requests
from diffusers import StableDiffusionPipeline

# ===== IMAGE GENERATOR SETUP =====
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)


# ===== GOOGLE GEMINI SETUP =====
API_KEY = 'AIzaSyDYB8B6YbArvH8U9e_BZfhPMFjbmgmcS0w'  # replace with your key
MODEL = 'gemini-2.0-flash'
API_URL = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}'

# ===== IMAGE GENERATION VIEW =====
def generate_image(request):
    if request.method == "POST":
        prompt = request.POST.get("prompt")
        steps = request.POST.get("steps", "50")

        if not prompt:
            return render(request, "generate.html", {"error": "Please enter a prompt."})

        try:
            steps = int(steps)
        except ValueError:
            steps = 50

        steps = max(1, min(steps, 1000))

        # generate image
        image = pipe(prompt, num_inference_steps=steps).images[0]

        # save to static folder
        filename = f"{uuid.uuid4().hex}.png"
        save_path = os.path.join("static", filename)
        image.save(save_path)

        static_url = f"/static/{filename}"

        # also encode to base64 (optional)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        return render(request, "result.html", {
            "prompt": prompt,
            "steps": steps,
            "image_static_url": static_url,
            "image_base64": img_base64
        })

    return render(request, "generate.html")


# ===== CHATBOT VIEW =====
def chatbot(request):
    # On GET, start fresh
    if request.method == "GET":
        request.session["chat_history"] = []

    chat_history = request.session.get("chat_history", [])

    if request.method == "POST":
        user_input = request.POST.get("prompt")
        if user_input:
            data = {
                "contents": [
                    {
                        "parts": [{"text": user_input}]
                    }
                ]
            }
            try:
                response = requests.post(API_URL, json=data)
                if response.status_code == 200:
                    result = response.json()
                    bot_reply = result['candidates'][0]['content']['parts'][0]['text']
                else:
                    bot_reply = f"Error: {response.status_code} — {response.text}"
            except Exception as e:
                bot_reply = f"Exception: {e}"

            chat_history.append({"user": user_input, "bot": bot_reply})
            request.session["chat_history"] = chat_history

    return render(request, "chatbot.html", {"chat_history": chat_history})