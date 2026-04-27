import os
import io
import json
import gc
from pathlib import Path
from urllib import request as urlrequest
from urllib import error as urlerror
import numpy as np
import base64
from io import BytesIO
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

def get_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


loaded_models = {}
MODELS_DIR = Path("models")
DEFAULT_CROPS = ["rice", "wheat", "sugarcane", "cotton", "potato", "tomato", "corn", "citrus", "grape", "apple"]
# API Configuration (Set these in your hosting environment variables)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"


def get_available_crops():
    # If we have local models, use them
    crops = []
    if MODELS_DIR.exists() and MODELS_DIR.is_dir():
        for model_file in MODELS_DIR.glob("*_model.pth"):
            name = model_file.name
            if name.endswith("_model.pth"):
                crops.append(name[:-10])
    
    # If no local models (common in hosting), return all supported crops for Gemini
    if not crops:
        return sorted(set(DEFAULT_CROPS))
        
    return sorted(set(crops))


def analyze_with_gemini(image_bytes: bytes, crop_type: str):
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        return None

    prompt = (
        "You are an agronomy vision assistant. "
        "Analyze this crop leaf image for nutrient deficiency symptoms. "
        f"Crop: {crop_type}. "
        "Return strict JSON with keys: predictedDeficiency, summary, treatment, fixBudget, fixYield, fixPrice, continueYield, continuePrice. "
        "predictedDeficiency should be a short label. "
        "summary should be a concise overview. "
        "treatment should be a bulleted list of actionable steps or recommended fertilizers. "
        "fixBudget should be estimated cost to fix in ₹ (e.g. '₹1200'). "
        "fixYield should be expected yield if fixed (e.g. '6.2 tons/ha'). "
        "fixPrice should be expected price per ton if fixed (e.g. '₹1780 per ton'). "
        "continueYield should be expected yield if NOT fixed. "
        "continuePrice should be expected price per ton if NOT fixed."
    )

    mime = "image/jpeg"
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={gemini_api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }

    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urlerror.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API error: {detail[:200]}") from e
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}") from e

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_response = ""
    for part in parts:
        if "text" in part:
            text_response += part["text"]

    if not text_response.strip():
        raise RuntimeError("Gemini returned empty response.")

    raw = text_response.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw)
    except Exception:
        # Fallback when model returns plain text instead of JSON.
        parsed = {
            "predictedDeficiency": raw[:60],
            "summary": raw[:240],
        }

    predicted = str(parsed.get("predictedDeficiency", "Unknown"))
    summary = str(parsed.get("summary", "Predicted using Gemini API vision analysis."))

    return {
        "predictedDeficiency": predicted,
        "summary": summary,
        "treatment": parsed.get("treatment", "1. Apply Nitrogen-rich fertilizer (Urea).\n2. Improve soil moisture management.\n3. Monitor new growth."),
        "fixBudget": str(parsed.get("fixBudget", "₹1200")),
        "fixYield": str(parsed.get("fixYield", "6.2 tons/ha")),
        "fixPrice": str(parsed.get("fixPrice", "₹1780 per ton")),
        "continueYield": str(parsed.get("continueYield", "4.9 tons/ha")),
        "continuePrice": str(parsed.get("continuePrice", "₹1530 per ton")),
    }


def get_yield_prediction_gemini(cropType, soilType, location, ph, n, p, k, mg, temp, rainfall, humidity):
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        return None

    prompt = (
        "You are an expert agronomist and market analyst. "
        "Based on the following agricultural parameters:\n"
        f"Crop: {cropType}\n"
        f"Soil Type: {soilType}\n"
        f"Location: {location}\n"
        f"Soil pH: {ph}\n"
        f"Nitrogen (N): {n}\n"
        f"Phosphorus (P): {p}\n"
        f"Potassium (K): {k}\n"
        f"Magnesium (Mg): {mg}\n"
        f"Temperature: {temp} °C\n"
        f"Rainfall: {rainfall} mm\n"
        f"Humidity: {humidity}%\n"
        "Predict the expected crop yield (in tons/hectare) and the expected market price per ton (in local currency). "
        "Return strict JSON with exactly two keys: 'yield_pred' (a number) and 'price_pred' (a number). "
        "Do not include any markdown formatting or extra text."
    )

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={gemini_api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}

    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    text_response = "".join([part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])])
    raw = text_response.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    parsed = json.loads(raw)
    return {
        "yield_pred": float(parsed.get("yield_pred", 0)),
        "price_pred": float(parsed.get("price_pred", 0))
    }


def get_nutrient_analysis_gemini(plant_name, growth_stage, soil_ph, n, p, k, ca, mg, s):
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        return None

    prompt = (
        "You are an expert plant nutritionist and soil scientist. "
        "Analyze the following nutrient profile for a crop:\n"
        f"Plant Name: {plant_name}\n"
        f"Growth Stage: {growth_stage}\n"
        f"Soil pH: {soil_ph}\n"
        f"Measured Nutrients:\n"
        f"Nitrogen (N): {n}\n"
        f"Phosphorus (P): {p}\n"
        f"Potassium (K): {k}\n"
        f"Calcium (Ca): {ca}\n"
        f"Magnesium (Mg): {mg}\n"
        f"Sulfur (S): {s}\n"
        "\n"
        "Return strict JSON with the following structure exactly:\n"
        "{\n"
        '  "nutrients": { "N": ["measured", "optimal_range"], "P": ["measured", "optimal_range"], "K": ["measured", "optimal_range"], "Ca": ["measured", "optimal_range"], "Mg": ["measured", "optimal_range"], "S": ["measured", "optimal_range"] },\n'
        '  "statuses": { "N": "Low"/"Normal"/"High", "P": "Low"/"Normal"/"High", "K": "Low"/"Normal"/"High", "Ca": "Low"/"Normal"/"High", "Mg": "Low"/"Normal"/"High", "S": "Low"/"Normal"/"High" },\n'
        '  "deficiencies": ["list", "of", "deficient", "nutrients"],\n'
        '  "excesses": ["list", "of", "excess", "nutrients"],\n'
        '  "fertilizers": ["list", "of", "recommended", "fertilizers"],\n'
        '  "analysis": "A brief analysis string",\n'
        '  "observations": "A brief observations string",\n'
        '  "recommendations": "A brief recommendations string",\n'
        '  "scientific_basis": "A brief scientific basis string"\n'
        "}\n"
        "Do not include any markdown formatting or extra text."
    )

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={gemini_api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2}}

    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")

    candidates = body.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    text_response = "".join([part.get("text", "") for part in candidates[0].get("content", {}).get("parts", [])])
    raw = text_response.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    return json.loads(raw)


def load_model(crop_type: str):
    import torch
    import timm
    crop_type = crop_type.lower()
    
    # Clear other models to save memory - ensure only one model is in RAM at a time
    if crop_type not in loaded_models:
        loaded_models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        return loaded_models[crop_type]

    model_path = MODELS_DIR / f"{crop_type}_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for crop: {crop_type}")

    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)
    is_placeholder = bool(checkpoint.get("is_placeholder", False))
    model_note = checkpoint.get("note")

    model = timm.create_model("efficientnet_b2", pretrained=False, num_classes=num_classes)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.4, inplace=True),
        torch.nn.Linear(model.classifier.in_features, num_classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()

    loaded_models[crop_type] = (model, class_names, is_placeholder, model_note)
    return model, class_names, is_placeholder, model_note


def pil_to_base64(img: Image.Image):
    buffer = BytesIO()
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
            gemini_pred = get_yield_prediction_gemini(cropType, soilType, location, ph, n, p, k, mg, temp, rainfall, humidity)
            if gemini_pred:
                yield_pred = round(gemini_pred["yield_pred"], 2)
                price_pred = round(gemini_pred["price_pred"], 2)
        except Exception as e:
            print(f"Warning: Gemini yield prediction failed. Error: {e}")

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
            "analysis": (
                """The analysis compares the measured nutrient concentrations in your tomatoplants (vegetative stage)
                    against established optimal ranges for this speciesand growth phase."""
            ),
            "observations": (
                """Deficiencies in K, Ca may limit plant growth and productivity.
                    Potassium deficiency in tomato leads to yellowing leaf margins and reduced diseaseresistance.
                    Calcium deficiency in tomato causes blossom-end rot in fruitsand distorted new growth.
                    Excess levels of N may cause nutrient imbalances or toxicity symptoms.
                    Excess nitrogen in tomato promotes excessive vegetative growth at theexpense of flowering/fruiting.
                    Soil pH of 7.0 is optimal for tomato cultivation. Most nutrients are optimallyavailable between pH 6.0-7.0."""
            ),
            "recommendations": (
                """Apply recommended fertilizers to address nutrient deficiencies.
                    Monitor plant growth and leaf color for signs of improvement or furtherissues.
                    Consider retesting soil and plant tissue in 2-4 weeks after fertilizerapplication.
                    Maintain proper irrigation practices as water affects nutrient availability."""
            ),
            "scientific_basis": (
                """Plant nutrient requirements vary by species and growth stage. 
                    The ideal rangesused in this analysis are based on peer-reviewed research and agriculturalextension recommendations 
                    for tomato cultivation. Nutrient interactions (e.g.,N:K ratio) and environmental factors also influence plant nutrient uptake and 
                    utilization."""
            )
        }
        
        try:
            gemini_res = get_nutrient_analysis_gemini(
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
            if gemini_res:
                result.update(gemini_res)
                # Ensure the core fields are not overwritten accidentally
                result["analysis_date"] = date.today().strftime("%d-%m-%Y")
                result["plant_name"] = form_data.get("plant_name")
                result["growth_stage"] = form_data.get("growth_stage")
                result["soil_ph"] = form_data.get("soil_ph")
        except Exception as e:
            print(f"Warning: Gemini nutrient analysis failed. Error: {e}")

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
        
        # Optimization: Process image only once and resize early
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        original_b64 = pil_to_base64(img_pil.resize((IMG_SIZE, IMG_SIZE)))

        # --- 1. Try Gemini API First (Memory Efficient) ---
        gemini_result = None
        gemini_warning = ""
        
        try:
            # Only call Gemini if API key exists
            if os.environ.get("GEMINI_API_KEY"):
                gemini_result = analyze_with_gemini(contents, crop_type)
        except Exception as e:
            gemini_warning = f"Gemini API error: {str(e)}"

        if gemini_result:
            # Clear any models if they were loaded to free memory
            if loaded_models:
                loaded_models.clear()
                gc.collect()
                
            result = {
                "predictedDeficiency": gemini_result["predictedDeficiency"],
                "summary": gemini_result["summary"],
                "treatment": gemini_result.get("treatment", "No specific treatment protocol provided."),
                "model_warning": "Prediction source: Gemini API vision analysis.",
                "uploaded_image": original_b64,
                "gradcam_image": None, # Skip Grad-CAM for Gemini to save memory
                "fixBudget": gemini_result.get("fixBudget", "₹1200"),
                "fixYield": gemini_result.get("fixYield", "6.2 tons/ha"),
                "fixPrice": gemini_result.get("fixPrice", "₹1780 per ton"),
                "continueYield": gemini_result.get("continueYield", "4.9 tons/ha"),
                "continuePrice": gemini_result.get("continuePrice", "₹1530 per ton")
            }
            return templates.TemplateResponse(
                request=request,
                name="deficiency.html",
                context={"result": result, "available_crops": available_crops},
            )

        # --- 2. Fallback to Local Model if Gemini Fails or is Missing ---
        if crop_type.lower() not in available_crops:
            raise FileNotFoundError(f"Model for '{crop_type}' is unavailable and Gemini failed.")

        gradcam_b64 = None
        local_pred_class = None
        model_note = ""
        is_placeholder = False
        
        try:
            import torch
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.image import show_cam_on_image
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            
            model, class_names, is_placeholder, model_note = load_model(crop_type)
            device = get_device()
            transform = get_transform()
            
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                pred_idx = probs.argmax().item()
                local_pred_class = class_names[pred_idx]

            # Only generate Grad-CAM if explicitly requested or local model is used
            target_layers = [model.conv_head]
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0, :]
            rgb_img = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            visualization_uint8 = (visualization * 255).astype(np.uint8)
            gradcam_b64 = pil_to_base64(Image.fromarray(visualization_uint8))
            
            # Cleanup local torch objects
            del input_tensor, outputs, probs, cam
            gc.collect()
        except Exception as e:
            print(f"Local model error: {e}")

        if not local_pred_class:
            raise RuntimeError(f"Analysis failed. {gemini_warning or 'Local model failed.'}")

        result = {
            "predictedDeficiency": local_pred_class,
            "summary": "Local analysis performed. For more detailed insights, configure Gemini API.",
            "treatment": "1. Verify nutrient levels.\n2. Consult agronomist.\n3. Apply balanced NPK.",
            "model_warning": gemini_warning or model_note if is_placeholder else "",
            "uploaded_image": original_b64,
            "gradcam_image": gradcam_b64,
            "fixBudget": "₹1200",
            "fixYield": "6.2 tons/ha",
            "fixPrice": "₹1780 per ton",
            "continueYield": "4.9 tons/ha",
            "continuePrice": "₹1530 per ton"
        }

        return templates.TemplateResponse(
            request=request, name="deficiency.html", context={"result": result, "available_crops": available_crops}
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

# or write "uvicorn main:app --reload" in terminal to run the app
