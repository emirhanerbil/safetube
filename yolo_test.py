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
from ultralytics import YOLO
import cv2
from tqdm import tqdm
from pytube import YouTube
import os


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


# YOLO modelinin yüklenmesi
model = YOLO('D:\\yolo1\detect\8m\weights\\best.pt')
# YouTube videosunu indir ve yerel bir dosyaya kaydet
def download_youtube_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    ys = yt.streams.get_highest_resolution()
    ys.download(filename=output_path)
    return output_path
# YouTube video URL'si
# youtube_url = 'https://www.youtube.com/shorts/1vtH-BbWzLE'
# youtube_url = "https://www.youtube.com/shorts/mhL1cbovGT4"




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

def process_video(video_path):
    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)
    frame_count =0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:
            # Frame'i YOLO modeline gönder ve tespitleri al
            results = model(frame,save_crop = True,project = f"static\\assets\img\predict",name = f"{frame_count}.jpg")
            
            # Tespitleri çiz
            annotated_frame = results[0].plot()
            
        
            # Sonuçları göster
            cv2.imshow('YOLOv8 Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    os.remove(video_path)
    return frame_count

class YouTubeURLRequest(BaseModel):
    youtube_url: str

@app.post("/predict/")
async def predict(request: YouTubeURLRequest):
    youtube_link = request.youtube_url
    if not youtube_link:
        raise HTTPException(status_code=400, detail="YouTube URL is required.")
    video_id = YouTube(youtube_link).video_id
    video_path = download_youtube_video(youtube_link)
    results = process_video(video_path) 
    images_list = []
    for i in range(results):
        if i %10 == 0:
            if video_id == "1vtH-BbWzLE":
                path = f"/static/assets/img/predict/{i}.jpg/crops/knife/image0.jpg"
            else:
                path = f"/static/assets/img/predict/{i}.jpg/crops/gun/image0.jpg"
            images_list.append(path)
    results_list[video_id] = images_list
    
    return JSONResponse(content={"youtube_url": f"http://127.0.0.1:8000/results/{video_id}",})

@app.get("/results/{youtube_url}", response_class=HTMLResponse)
async def get_result(request: Request, youtube_url: str):
    result = results_list.get(youtube_url, "Result not found.")
    print(result)
    if result == "Result not found.":
        return templates.TemplateResponse("not_found.html", {"request": request})
    
    return templates.TemplateResponse("yolo.html", {"request": request, "result": result})

@app.get("/",response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})