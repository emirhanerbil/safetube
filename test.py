from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import keras
from pydantic import BaseModel
import urllib.parse
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pytube import YouTube
from fastapi.middleware.cors import CORSMiddleware

# Modeli yükleme
# path = "bitirme.keras"
# model = keras.saving.load_model(path)
# model.summary()

# #Tokenizer'ı yükleme
# with open("bitirme.json", "r") as f:
#     tokenizer = json.load(f)

# tokenizer = tokenizer_from_json(tokenizer)
import pickle

with open('bitirme_model.pickle', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

origins = [
    "chrome-extension://mkiplihndmffgfnefmbopejpdjjdofjl",
    "http://localhost:8000",
    "chrome-extension://fodpildcedcfegocbfbjlalhmmlkgfdc"
    # Diğer izin verilen origin'leri buraya ekleyin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Bellekte sonuçları saklamak için bir sözlük
results_list = {}

def get_caption(captions):
    sentiment_list = []
    for obj in captions.json_captions["events"]:
        sentiment_obj = {}
        texts = obj.get("segs",[None])
        if texts[0]:
            sentiment = ""
            if obj.get("segs",[None])[0]["utf8"] == "\n":
                continue
            for text in texts:
                if text.get("utf8","none") == "\n":
                    continue
                if "[\xa0__\xa0]" in text.get("utf8","none"):
                    text["utf8"] = text["utf8"].replace("[\xa0__\xa0]","amk")
                sentiment += text.get("utf8","none") + " "

            start = obj["tStartMs"]
            end = start + obj["dDurationMs"]
            sentiment_obj["sentiment"] = sentiment
            sentiment_obj["start"] = start/1000
            sentiment_obj["end"] = end/1000
            sentiment_list.append(sentiment_obj)
    return sentiment_list

def results(sentiment_list,link):
    link_list = []
    for obj in sentiment_list:
        link_obj = {}
        text = obj["sentiment"]
        text= text.replace("İ","i").replace("I","ı").lower()
        # text = tokenizer.texts_to_sequences([text])
        # text = pad_sequences(text, maxlen=1135)
        # prediction = model.predict(text)
        # result =  prediction.argmax()
            # if sen["text"] == "kız":
            #     print(sen["text"])
        result = model.predict([text])
        if result != "OTHER" and result != "INSULT":
            link_obj["text"] = obj["sentiment"]
            print(link_obj["text"])
            if "kız" in link_obj["text"]:
                print("girdi")
                continue
            # link_obj["result"] = int(result)
            link_obj["result"] = result
            time_stamp = int(obj["start"])
            link_obj["link"] = f"{link}&t={time_stamp}s"
            link_list.append(link_obj)
    return link_list

# Yapay zeka modelini simüle eden basit bir fonksiyon
def ai_model_predict(youtube_url):
    yt = YouTube(youtube_url)
    stream = yt.streams.get_audio_only()
    try:
        captions = yt.captions["tr"]
    except KeyError as error:
        print("auto_generated",error,youtube_url)
        captions = yt.captions["a.tr"]
    except:
        captions = "not_found"
        
    sentiment_list = get_caption(captions)
    sentiment_list_with_results = results(sentiment_list,youtube_url)
    return sentiment_list_with_results

class YouTubeURLRequest(BaseModel):
    youtube_url: str

@app.post("/predict/")
async def predict(request: YouTubeURLRequest):
    youtube_link = request.youtube_url
    if not youtube_link:
        raise HTTPException(status_code=400, detail="YouTube URL is required.")
    video_id = YouTube(youtube_link).video_id
    if video_id in results_list:
        print("Önceden Kontrol Edilmiş.")
        return JSONResponse(content={"youtube_url": f"http://18.194.92.133/results/{video_id}",})
    
    prediction = ai_model_predict(youtube_link)
    # Sonucu sakla
    results_list[video_id] = prediction
    
    return JSONResponse(content={"youtube_url": f"http://18.194.92.133/results/{video_id}",})

@app.get("/results/{youtube_url}", response_class=HTMLResponse)
async def get_result(request: Request, youtube_url: str):
    result = results_list.get(youtube_url, "Result not found.")
    print(result)
    if result == "Result not found.":
        return templates.TemplateResponse("not_found.html", {"request": request})
    
    return templates.TemplateResponse("result_page.html", {"request": request, "result": result})

@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.get("/privacy",response_class=HTMLResponse)
async def privacy_page(request: Request):
    return templates.TemplateResponse("privacy.html",{"request": request})