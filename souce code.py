import speech_recognition as sr # recognise speech
import playsound # to play an audio files
from gtts import gTTS # google text to speech
import random
from time import ctime # get time details
import webbrowser # open browser
import ssl
import certifi
import time
import os # to remove created audio files
from PIL import Image
import subprocess
import pyautogui #screenshot
import pyttsx3
import bs4 as bs
import urllib.request
import cv2
#Ai libraries
from imageai.Detection import ObjectDetection
import threading 
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from utils import *
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
import pandas as pd
from sklearn.externals import joblib
from PIL import Image
import numpy as np
import urllib.request
import urllib.parse
import keras
from keras.models import load_model
from keras.preprocessing import image as im

from skimage import color
from skimage import io
# img = color.rgb2gray(io.imread('image.png'))
import pytesseract
# ---------------------------

import tensorflow as tf
global graph,model2,autoencoder

max_val = 8
max_pt = -1
max_kp = 0
orb = cv2.ORB_create()

class person:
    name = ''
    def setName(self, name):
        self.name = name

class asis:
    name = ''
    def setName(self, name):
        self.name = name



def there_exists(terms):
    for term in terms:
        if term in voice_data:
            return True

def engine_speak(text):
    text = str(text)
    engine.say(text)
    engine.runAndWait()

r = sr.Recognizer() # initialise a recogniser
# listen for audio and convert it to text:
def record_audio(ask=""):
    with sr.Microphone() as source: # microphone as source
        if ask:
            engine_speak(ask)
        audio = r.listen(source, 5, 5)  # listen for the audio via source
        print("Done Listening")
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)  # convert audio to text
        except sr.UnknownValueError: # error: recognizer does not understand
            engine_speak('I did not get that')
        except sr.RequestError:
            engine_speak('Sorry, the service is down') # error: recognizer is not connected
        print(">>", voice_data.lower()) # print what user said
        return voice_data.lower()

# get string and make a audio file to be played
def engine_speak(audio_string):
    audio_string = str(audio_string)
    tts = gTTS(text=audio_string, lang='en') # text to speech(voice)
    r = random.randint(1,20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file) # save as mp3
    playsound.playsound(audio_file) # play the audio file
    print(asis_obj.name + ":", audio_string) # print what app said
    os.remove(audio_file) # remove audio file

def respond(voice_data):
    # 1: greeting
    if there_exists(['hey','hi','hello']):
        greetings = ["hey, how can I help you" + person_obj.name, "hey, what's up?" + person_obj.name, "I'm listening" + person_obj.name, "how can I help you?" + person_obj.name, "hello" + person_obj.name]
        greet = greetings[random.randint(0,len(greetings)-1)]
        engine_speak(greet)

    # 2: name
    if there_exists(["what is your name","what's your name","tell me your name"]):
        if person_obj.name:
            engine_speak("whats with my name ")
        else:
            engine_speak("i dont know my name . what's your name?")

    if there_exists(["my name is"]):
        person_name = voice_data.split("is")[-1].strip()
        engine_speak("okay, i will remember that " + person_name)
        person_obj.setName(person_name) # remember name in person object
    
    if there_exists(["your name should be"]):
        asis_name = voice_data.split("be")[-1].strip()
        engine_speak("okay, i will remember that my name is " + asis_name)
        asis_obj.setName(asis_name) # remember name in asis object

    # 3: greeting
    if there_exists(["how are you","how are you doing"]):
        engine_speak("I'm very well, thanks for asking " + person_obj.name)

    # 4: time
    if there_exists(["what's the time","tell me the time","what time is it"]):
        time = ctime().split(" ")[3].split(":")[0:2]
        if time[0] == "00":
            hours = '12'
        else:
            hours = time[0]
        minutes = time[1]
        time = hours + " hours and " + minutes + "minutes"
        engine_speak(time)

    # 5: search google
    if there_exists(["search for"]) and 'youtube' not in voice_data:
        search_term = voice_data.split("for")[-1]
        url = "https://google.com/search?q=" + search_term
        webbrowser.get().open(url)
        engine_speak("Here is what I found for" + search_term + "on google")

    # 6: search youtube
    if there_exists(["youtube"]):
        search_term = voice_data.split("for")[-1]
        url = "https://www.youtube.com/results?search_query=" + search_term
        webbrowser.get().open(url)
        engine_speak("Here is what I found for " + search_term + "on youtube")

    #7: get stock price
    if there_exists(["price of"]):
        search_term = voice_data.split("for")[-1]
        url = "https://google.com/search?q=" + search_term
        webbrowser.get().open(url)
        engine_speak("Here is what I found for " + search_term + " on google")
    
    # search for music
    if there_exists(["play music"]):
        search_term= voice_data.split("for")[-1]
        url="https://open.spotify.com/search/"+search_term
        webbrowser.get().open(url)
        engine_speak("You are listening to"+ search_term +"enjoy sir")
    #search for amazon.com
    if there_exists(["amazon.com"]):
        search_term = voice_data.split("for")[-1]
        url="https://www.amazon.in"+search_term
        webbrowser.get().open(url)
        engine_speak("here is what i found for"+search_term + "on amazon.com")
         
    #make a note
    if there_exists(["make a note"]):
        search_term=voice_data.split("for")[-1]
        url="https://keep.google.com/#home"
        webbrowser.get().open(url)
        engine_speak("Here you can make notes")
        
    #open instagram
    if there_exists(["open instagram","want to have some fun time"]):
        search_term=voice_data.split("for")[-1]
        url="https://www.instagram.com/"
        webbrowser.get().open(url)
        engine_speak("opening instagram")
        
    #open twitter
    if there_exists(["open twitter"]):
        search_term=voice_data.split("for")[-1]
        url="https://twitter.com/"
        webbrowser.get().open(url)
        engine_speak("opening twitter")
        
    #8 time table
    if there_exists(["show my time table"]):
        im = Image.open(r"D:\WhatsApp Image 2019-12-26 at 10.51.10 AM.jpeg")
        im.show()
    
    #9 weather
    if there_exists(["weather","tell me the weather report","whats the condition outside"]):
        search_term = voice_data.split("for")[-1]
        url = "https://www.google.com/search?sxsrf=ACYBGNSQwMLDByBwdVFIUCbQqya-ET7AAA%3A1578847393212&ei=oUwbXtbXDN-C4-EP-5u82AE&q=weather&oq=weather&gs_l=psy-ab.3..35i39i285i70i256j0i67l4j0i131i67j0i131j0i67l2j0.1630.4591..5475...1.2..2.322.1659.9j5j0j1......0....1..gws-wiz.....10..0i71j35i39j35i362i39._5eSPD47bv8&ved=0ahUKEwiWrJvwwP7mAhVfwTgGHfsNDxsQ4dUDCAs&uact=5"
        webbrowser.get().open(url)
        engine_speak("Here is what I found for on google")
    
    #open gmail
    if there_exists(["open my mail","gmail","check my email"]):
        search_term = voice_data.split("for")[-1]
        url="https://mail.google.com/mail/u/0/#inbox"
        webbrowser.get().open(url)
        engine_speak("here you can check your gmail")
    

    #10 stone paper scisorrs
    
    if there_exists(["game"]):
        voice_data = record_audio("choose among rock paper or scissor")
        moves=["rock", "paper", "scissor"]
    
        cmove=random.choice(moves)
        pmove=voice_data
        

        engine_speak("The computer chose " + cmove)
        engine_speak("You chose " + pmove)
        #engine_speak("hi")
        if pmove==cmove:
            engine_speak("the match is draw")
        elif pmove== "rock" and cmove== "scissor":
            engine_speak("Player wins")
        elif pmove== "rock" and cmove== "paper":
            engine_speak("Computer wins")
        elif pmove== "paper" and cmove== "rock":
            engine_speak("Player wins")
        elif pmove== "paper" and cmove== "scissor":
            engine_speak("Computer wins")
        elif pmove== "scissor" and cmove== "paper":
            engine_speak("Player wins")
        elif pmove== "scissor" and cmove== "rock":
            engine_speak("Computer wins")

    #11 toss a coin
    if there_exists(["toss","flip","coin"]):
        moves=["head", "tails"]   
        cmove=random.choice(moves)
        engine_speak("The computer chose " + cmove)

    #12 calc
    if there_exists(["plus","minus","multiply","divide","power","+","-","*","/"]):
        opr = voice_data.split()[1]

        if opr == '+':
            engine_speak(int(voice_data.split()[0]) + int(voice_data.split()[2]))
        elif opr == '-':
            engine_speak(int(voice_data.split()[0]) - int(voice_data.split()[2]))
        elif opr == 'multiply':
            engine_speak(int(voice_data.split()[0]) * int(voice_data.split()[2]))
        elif opr == 'divide':
            engine_speak(int(voice_data.split()[0]) / int(voice_data.split()[2]))
        elif opr == 'power':
            engine_speak(int(voice_data.split()[0]) ** int(voice_data.split()[2]))
        else:
            engine_speak("Wrong Operator")
        
    #13 screenshot
    if there_exists(["capture","my screen","screenshot"]):
        myScreenshot = pyautogui.screenshot()
        myScreenshot.save('C:/Users/YASH/Pictures/Screenshots') 
    
    
    #14 to search wikipedia for definition
    if there_exists(["definition of"]):
        definition=record_audio("what do you need the definition of")
        url=urllib.request.urlopen('https://en.wikipedia.org/wiki/'+definition)
        soup=bs.BeautifulSoup(url,'lxml')
        definitions=[]
        for paragraph in soup.find_all('p'):
            definitions.append(str(paragraph.text))
        if definitions:
            if definitions[0]:
                engine_speak('im sorry i could not find that definition, please try a web search')
            elif definitions[1]:
                engine_speak('here is what i found '+definitions[1])
            else:
                engine_speak ('Here is what i found '+definitions[2])
        else:
                engine_speak("im sorry i could not find the definition for "+definition)


    if there_exists(["exit", "quit", "goodbye"]):
        engine_speak("we could continue more sir, but.,,...,,,,,..,,,,, byee")
        exit()


time.sleep(1)

person_obj = person()
asis_obj = asis()
asis_obj.name = 'Kim'
engine = pyttsx3.init()


while(1):
    voice_data = record_audio("Recording") # get the voice input
    print("Done")
    print("Q:", voice_data)
    respond(voice_data) # respond


class VideoCamera(object):    
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path , "yolo-tiny.h5")) 
    detector.loadModel(detection_speed="flash")
    
    #buffer
    All_faces = [0,0,0,0,0,0]
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()    
        
    #queue
    def second(self,name): 
        self.All_faces[self.All_faces.index(name)] = 0
    
    #text-to-speech
    def speak(self,detections):
        for eachObject in detections:
            name = eachObject["name"]
            if name not in self.All_faces:
                if (self.All_faces[0] != 0):
                    for i in range(5,0,-1):
                        self.All_faces[i]=self.All_faces[i-1]
                self.All_faces[0]=name

                timer = threading.Timer(20, self.second,args=[name]) 
                timer.start()

                engine = pyttsx3.init()
                engine.say('I see '+name)
                engine.runAndWait()

    #detection
    def get_frame(self,flag): 
        
        if flag==1:
            engine = pyttsx3.init()
            engine.say('starting object detetction live stream.')
            engine.runAndWait()
            self.video = cv2.VideoCapture(0)
            
        ret, frame = self.video.read()
        
        detected_image_array, detections = self.detector.detectObjectsFromImage(input_type="array", input_image=frame, output_type="array",minimum_percentage_probability=30)
        
        self.speak(detections)

        ret, jpeg = cv2.imencode('.jpg', detected_image_array)
        return jpeg.tobytes()
    
    def switching(self):
        if(self.video.isOpened()):
            engine = pyttsx3.init()
            engine.say('switching to face recognition')
            engine.runAndWait()
            self.video.release()
    
    #closing
    def close(self):
        engine = pyttsx3.init()
        engine.say('stopping live stream.')
        engine.runAndWait()
        self.video.release()
        
# detection of currency notes
graph = tf.get_default_graph()

app = Flask(__name__)
model2 = load_model('final.h5')    #Lenet-aw.h5

autoencoder = load_model('happyHack.h5')

# model = keras.Sequential()
# model = load_model('LeNetArch.h5')
# model = load_model('litemodel.sav')
# model = joblib.load('lenet-aw.h5')   
# 'NoteDetection.h5'  'litemodel.sav'  LeNetArch.h5

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get():
	input = dict(request.form)
	# accidentindex = input['accidentindex'][0]
	# age_of_vehicle = input['age_of_vehicle'][0]

	# if request.method == 'POST':  
	#     f = request.files['file']  
	#     f.save(f.filename) 

	# file = request.files['file']
	# if request.method == 'POST':  
	#     f = request.files['file']
	#     f.save(f.filename)  
	# filename= request.form['hidden']  #input['imgname']
	# filename = request.form.get('imgname')
	print("filename=",request.form.get('imgname'))  

	print("filename=",input['imgname'][0])   

	v= input['imgname'][0]
	#session
	# s = session['key']
	# print("s =",s) 
	 # --------------------------------------------------------------
	 # grey scale
	# img = color.rgb2gray(io.imread(v))
	# img = im.load_img(v, target_size=(256, 256))
	# img = color.rgb2gray(io.imread(img))

	# data = np.array([np.array(img)]) 
	# data = np.array([np.array(Image.open(v))])  #f.filename
	i=cv2.imread(v,0); # input as grey scale
	width = 256
	height = 256
	dim = (width, height)

	# resize image
	resized = cv2.resize(i, dim, interpolation = cv2.INTER_AREA)

	data = np.array([np.array(resized)])
	# -----------------------------------------------------


	# data = np.array([np.array(Image.open('500.jpg'))])  #/utilities/
	data = data.reshape(-1,256,256,1)
	data = data.astype('float32')
	data = data/ 255.
	# print("logging ",data)
	# return str(data);

	# model2._make_predict_function()

	with graph.as_default():
	 y = model2.predict(data, verbose=1)

	# try: y =model2.predict(data, verbose=1)
	# except Exception as e: y = str(e)

	# y =model2.predict(data, verbose=1)
	# y = y.astype('float32')
	# r = [round(x[0]) for x in y]
	# r = float(round(y[0][0]))
	# res = np.array(y, dtype='float64')
	# res = np.array(r, dtype='float64')
	print(" Y =  ",y)
	# print(" res =  ",res)
	# result = int(round(y[0][0])) 
	# https://thispointer.com/find-max-value-its-index-in-numpy-array-numpy-amax/
	result = np.where(y[0] == np.amax(y[0]))
	print("result[0]=",result[0])
	print("result[0][0]=",result[0][0])
	pre=result[0][0]
	# ------------------------
	# return str(result[0][0])

	# -------------------serial number -----------
	img2 = cv2.imread(v, 0)
	# cropped=img2[115:130, 30:100]
	if pre == 0:
	    cropped=img2[115:130, 30:100]     # 10-sns (30,115) (100,130)
	elif pre == 1:
	    cropped=img2[250:300, 70:265]     # 20-sns (70,250) (265,300)
	elif pre == 2:    
	    cropped=img2[450:520, 130:430] #100-sns
	elif pre == 3:    
	    cropped = img2[150:200, 75:390] #500-sns    
	# cropped = img2[180:250, 80:550] #realmoney (75,150)(390 ,200)

	config = ("-l eng --oem 3 --psm 6")
	number = cv2.fastNlMeansDenoising(cropped, 10, 7,21)
	number[number > 170] = 255
	number[number <= 170] = 0

	# config = ("-l eng --oem 3 --psm 3")
	# kernel1 = np.ones((3, 3), np.uint8)
	# kernel2 = np.ones((5, 5), np.uint8)
	# number = cv2.morphologyEx(number, cv2.MORPH_CLOSE, kernel1)
	#number = cv2.morphologyEx(number, cv2.MORPH_OPEN, kernel2)
	# number = cv2.GaussianBlur(number, (3, 3), 0)       # For Smoothing

	text = pytesseract.image_to_string(number, config=config)
	print("text=",text)

	# text="7AH 433534"
	# --------------------counterfeit------------------
	# xtest = data

	xtest=cv2.imread(v, 0)
	xtest = cv2.resize(xtest,(256,256))
	xtest=xtest.reshape(-1,256,256,1)
	xtest = xtest.astype('float32')
	xtest = xtest/ 255.

	with graph.as_default():
	 decimg = autoencoder.predict(data, verbose=1)
	# decimg = autoencoder.predict(xtest)
	# xtest = data
	# decimg = autoencoder._make_predict_function(xtest)

	mse1 = np.mean(np.power(xtest - decimg, 2),axis=1)
	mse0 = np.mean(mse1, axis=1)
	print(mse0[0])
	res=mse0[0]
	con=0
	if res > 0.002 :
		if res < 0.01:
			con=1

	if con == 0:
		return "-0-----------";	
	# ----------------------------------
	#  (denomination - 0/1/2/3) + (counterfeit= 0/1) + (sno)
	ret = str(result[0][0]) + str("1") + text;
	print("ret= ",ret);
	return ret;
    
    

# @app.route('/success', methods = ['POST'])  
# def success():  
#     if request.method == 'POST':  
#         f = request.files['file']  
#         f.save(f.filename)  
#         return render_template("index.html", name = f.filename)  

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4000)
