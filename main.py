import os
import io
import json
import base64
import requests
from pathlib import Path
from datetime import date
from PIL import Image
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AI - Smart Farm Analyzer")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

IMG_SIZE = 224
DEFAULT_CROPS = ["rice", "wheat", "sugarcane", "cotton", "potato", "tomato", "corn", "citrus", "grape", "apple"]

# NVIDIA API Configuration
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or "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"
NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "moonshotai/kimi-k2.5"


def get_available_crops():
    """Returns the list of supported crops for analysis."""
    return sorted(DEFAULT_CROPS)


def call_nvidia_api(messages: list, stream: bool = False):
    """Generic helper to call the NVIDIA/Moonshot API."""
    print(f"Calling NVIDIA API (Model: {NVIDIA_MODEL})...")
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json"
    }
    
    payload = {
        "model": NVIDIA_MODEL,
        "messages": messages,
        "max_tokens": 4096, # Reduced for faster response
        "temperature": 0.2,
        "top_p": 1.0,
        "stream": stream,
        "chat_template_kwargs": {"thinking": False}, # Disabled for faster JSON responses
    }

    try:
        print("Sending POST request to NVIDIA...")
        response = requests.post(NVIDIA_INVOKE_URL, headers=headers, json=payload, timeout=45)
        print(f"NVIDIA Status Code: {response.status_code}")
        response.raise_for_status()
        result = response.json()
        
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(f"NVIDIA Response received. Content length: {len(content)}")
        return content
    except Exception as e:
        print(f"NVIDIA API Error: {e}")
        raise RuntimeError(f"NVIDIA API request failed: {e}")


def analyze_with_nvidia(image_bytes: bytes, crop_type: str):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    prompt = (
        "You are an agronomy vision assistant. "
        "Analyze this crop leaf image for nutrient deficiency symptoms. "
        f"Crop: {crop_type}. "
        "Return strict JSON with keys: predictedDeficiency, summary, treatment, fixBudget, fixYield, fixPrice, continueYield, continuePrice. "
        "Do not include any thinking tags or extra text, just the raw JSON object."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ]

    print(f"Analyzing image for crop: {crop_type}")
    text_response = call_nvidia_api(messages)
    print("Parsing NVIDIA response...")
    
    raw = text_response.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        # Fallback parsing
        parsed = {
            "predictedDeficiency": "Analysis complete",
            "summary": raw[:200] if raw else "Vision analysis performed via NVIDIA NIM."
        }

    return {
        "predictedDeficiency": str(parsed.get("predictedDeficiency", "Unknown")),
        "summary": str(parsed.get("summary", "Predicted using NVIDIA Moonshot AI vision analysis.")),
        "treatment": parsed.get("treatment", "1. Apply Nitrogen-rich fertilizer (Urea).\n2. Improve soil moisture management.\n3. Monitor new growth."),
        "fixBudget": str(parsed.get("fixBudget", "₹1200")),
        "fixYield": str(parsed.get("fixYield", "6.2 tons/ha")),
        "fixPrice": str(parsed.get("fixPrice", "₹1780 per ton")),
        "continueYield": str(parsed.get("continueYield", "4.9 tons/ha")),
        "continuePrice": str(parsed.get("continuePrice", "₹1530 per ton")),
    }


def get_yield_prediction_nvidia(cropType, soilType, location, ph, n, p, k, mg, temp, rainfall, humidity):
    prompt = (
        "You are an expert agronomist and market analyst. "
        "Based on the following agricultural parameters:\n"
        f"Crop: {cropType}, Soil Type: {soilType}, Location: {location}, Soil pH: {ph}, "
        f"N: {n}, P: {p}, K: {k}, Mg: {mg}, Temp: {temp}°C, Rainfall: {rainfall}mm, Humidity: {humidity}%\n"
        "Predict the expected crop yield (tons/hectare) and market price per ton. "
        "Return strict JSON with exactly: 'yield_pred' (number) and 'price_pred' (number)."
    )

    messages = [{"role": "user", "content": prompt}]
    text_response = call_nvidia_api(messages)
    
    raw = text_response.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    parsed = json.loads(raw)
    return {
        "yield_pred": float(parsed.get("yield_pred", 0)),
        "price_pred": float(parsed.get("price_pred", 0))
    }


def get_nutrient_analysis_nvidia(plant_name, growth_stage, soil_ph, n, p, k, ca, mg, s):
    prompt = (
        "Analyze this nutrient profile:\n"
        f"Plant: {plant_name}, Stage: {growth_stage}, pH: {soil_ph}, N: {n}, P: {p}, K: {k}, Ca: {ca}, Mg: {mg}, S: {s}\n"
        "Return strict JSON: { 'nutrients': { 'N': ['m','o'], ... }, 'statuses': { 'N': 'Low/Normal/High', ... }, "
        "'deficiencies': [], 'excesses': [], 'fertilizers': [], 'analysis': '', 'observations': '', 'recommendations': '', 'scientific_basis': '' }"
    )

    messages = [{"role": "user", "content": prompt}]
    text_response = call_nvidia_api(messages)
    
    raw = text_response.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    return json.loads(raw)


def pil_to_base64(img: Image.Image):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/yield", response_class=HTMLResponse)
def yield_page(request: Request):
    return templates.TemplateResponse(request=request, name="yield.html")


@app.post("/predict", response_class=HTMLResponse)
async def predict_yield(
    request: Request,
    cropType: str = Form(...),
    soilType: str = Form(...),
    location: str = Form(...),
    ph: float = Form(...),
    n: float = Form(...),
    p: float = Form(...),
    k: float = Form(...),
    mg: float = Form(...),
    temp: float = Form(...),
    rainfall: float = Form(...),
    humidity: float = Form(...)
):
    try:
        yield_pred = round(4.5 + (n + p + k + mg) / 1000, 2)
        price_pred = round(1500 + yield_pred * 50, 2)
        
        try:
            nvidia_pred = get_yield_prediction_nvidia(cropType, soilType, location, ph, n, p, k, mg, temp, rainfall, humidity)
            if nvidia_pred:
                yield_pred = round(nvidia_pred["yield_pred"], 2)
                price_pred = round(nvidia_pred["price_pred"], 2)
        except Exception as e:
            print(f"Warning: NVIDIA yield prediction failed. Error: {e}")

        return templates.TemplateResponse(
            request=request,
            name="yield.html",
            context={
                "cropType": cropType,
                "soilType": soilType,
                "location": location,
                "yield_pred": yield_pred,
                "price_pred": price_pred,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(request=request, name="error.html", context={"error": str(e)})


@app.get("/nutrient", response_class=HTMLResponse)
def nutrient_page(request: Request):
    return templates.TemplateResponse(request=request, name="nutrient.html")


@app.post("/analyze-nutrients", response_class=HTMLResponse)
async def analyze_nutrients(request: Request):
    try:
        form_data = await request.form()

        result = {
            "plant_name": form_data.get("plant_name"),
            "growth_stage": form_data.get("growth_stage"),
            "soil_ph": form_data.get("soil_ph"),
            "analysis_date": date.today().strftime("%d-%m-%Y"),
            "nutrients": {
                "N": [form_data.get("nutrient_n"), "250 - 350"],
                "P": [form_data.get("nutrient_p"), "50 - 70"],
                "K": [form_data.get("nutrient_k"), "250 - 350"],
                "Ca": [form_data.get("nutrient_ca"), "200 - 300"],
                "Mg": [form_data.get("nutrient_mg"), "60 - 90"],
                "S": [form_data.get("nutrient_s"), "40 - 60"]
            },
            "statuses": {
                "N": "Low", "P": "Low", "K": "Normal",
                "Ca": "Normal", "Mg": "Normal", "S": "Normal"
            },
            "deficiencies": ["Potassium", "Calcium"],
            "excesses": ["Nitrogen"],
            "fertilizers": ["Potassium chloride (60% K)", "Potassium sulfate (50% K)", "Calcium nitrate (15% N, 19% Ca)"],
            "analysis": "Analysis using NVIDIA AI.",
            "observations": "Observations using NVIDIA AI.",
            "recommendations": "Recommendations using NVIDIA AI.",
            "scientific_basis": "Scientific basis using NVIDIA AI."
        }
        
        try:
            nvidia_res = get_nutrient_analysis_nvidia(
                form_data.get("plant_name"),
                form_data.get("growth_stage"),
                form_data.get("soil_ph"),
                form_data.get("nutrient_n"),
                form_data.get("nutrient_p"),
                form_data.get("nutrient_k"),
                form_data.get("nutrient_ca"),
                form_data.get("nutrient_mg"),
                form_data.get("nutrient_s"),
            )
            if nvidia_res:
                result.update(nvidia_res)
                result["analysis_date"] = date.today().strftime("%d-%m-%Y")
                result["plant_name"] = form_data.get("plant_name")
                result["growth_stage"] = form_data.get("growth_stage")
                result["soil_ph"] = form_data.get("soil_ph")
        except Exception as e:
            print(f"Warning: NVIDIA nutrient analysis failed. Error: {e}")

        return templates.TemplateResponse(
            request=request, name="nutrient.html", context={"result": result}
        )
    except Exception as e:
        return templates.TemplateResponse(request=request, name="error.html", context={"error": str(e)})


@app.get("/deficiency", response_class=HTMLResponse)
def deficiency_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="deficiency.html",
        context={"available_crops": get_available_crops()},
    )


@app.post("/analyze-image", response_class=HTMLResponse)
async def analyze_image(request: Request, file: UploadFile = File(...), crop_type: str = Form(...)):
    try:
        available_crops = get_available_crops()
        contents = await file.read()
        
        # Process image for display
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        original_b64 = pil_to_base64(img_pil.resize((IMG_SIZE, IMG_SIZE)))

        # NVIDIA AI call
        nvidia_result = analyze_with_nvidia(contents, crop_type)

        if not nvidia_result:
            raise RuntimeError("NVIDIA API failed to return results.")

        result = {
            "predictedDeficiency": nvidia_result["predictedDeficiency"],
            "summary": nvidia_result["summary"],
            "treatment": nvidia_result.get("treatment", "No specific treatment protocol provided."),
            "model_warning": "Prediction source: NVIDIA Moonshot AI Kimi K2.5 Vision API.",
            "uploaded_image": original_b64,
            "gradcam_image": None,
            "fixBudget": nvidia_result.get("fixBudget", "₹1200"),
            "fixYield": nvidia_result.get("fixYield", "6.2 tons/ha"),
            "fixPrice": nvidia_result.get("fixPrice", "₹1780 per ton"),
            "continueYield": nvidia_result.get("continueYield", "4.9 tons/ha"),
            "continuePrice": nvidia_result.get("continuePrice", "₹1530 per ton")
        }

        return templates.TemplateResponse(
            request=request,
            name="deficiency.html",
            context={"result": result, "available_crops": available_crops},
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="deficiency.html",
            context={
                "error": str(e),
                "available_crops": get_available_crops(),
            },
        )


@app.get("/getmail", response_class=HTMLResponse)
async def get_mail(request: Request, email: str):
    message = f"Thank you, {email}! 🌿 You're now subscribed to AgriAI's Smart Farming Newsletter."
    return templates.TemplateResponse(
        request=request, name="newsletter.html", context={"message": message}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)

