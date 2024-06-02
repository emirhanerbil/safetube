from fastapi import FastAPI, Form, HTTPException, Query, Request
from typing import List
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.staticfiles import StaticFiles
from pytube import YouTube
import keras
from pytube.exceptions import VideoUnavailable,AgeRestrictedError,VideoPrivate,RegexMatchError
from typing import List, Dict
import numpy as np

# Modeli yükleme
path = "bitirme.keras"
model = keras.saving.load_model(path)
model.summary()

#Tokenizer'ı yükleme
with open("bitirme.json", "r") as f:
    tokenizer = json.load(f)

tokenizer = tokenizer_from_json(tokenizer)

app = FastAPI()
# Static files için ayar (örneğin, CSS, JS dosyaları)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Jinja2Templates ile HTML şablonlarını kullanmak için templates klasörünü belirtiyoruz
templates = Jinja2Templates(directory="templates")

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
        text= text.replace("İ","i").replace("I","ı").lower().strip()
        text = tokenizer.texts_to_sequences([text])
        text = pad_sequences(text, maxlen=1135)
        prediction = model.predict(text)
        result =  prediction.argmax()
        if result != 0:
            link_obj["text"] = obj["sentiment"]
            link_obj["result"] = str(np.int64(result))
            time_stamp = obj["start"]
            link_obj["link"] = f"{link}&t={time_stamp}s"
            link_list.append(link_obj)
    return link_list

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("landing.html", {"request": request})

@app.get("/process_param/")
async def process_param(param: str,response_model=List[Dict[str, str]]):
    
    yt = YouTube(param)
    stream = yt.streams.get_audio_only()
    try:
        captions = yt.captions["tr"]
    except KeyError as error:
        print("auto_generated",error,param)
        captions = yt.captions["a.tr"]
    except:
        captions = "not_found"
    
    sentiment_list = get_caption(captions)
    sentiment_list_with_results = results(sentiment_list,param)
    print(sentiment_list_with_results)
    
    return sentiment_list_with_results


