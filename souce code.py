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
        

# orb is an alternative to SIFT
#test_img = read_img('files/test_100_2.jpg')
test_img = read_img('files/test_50_1.jpg')
#test_img = read_img('files/download.png')
#test_img = read_img('files/test_100_3.jpg')
#test_img = read_img('files/test_20_4.jpg')
display('original1', test_img)
# resizing must be dynamic
original = resize_img(test_img, 0.4)
#display('original', original)
# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)
training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg',
'files/500.jpg']
for i in range(0, len(training_set)):
# train image
train_img = cv2.imread(training_set[i])
(kp2, des2) = orb.detectAndCompute(train_img, None)
# brute force matcher
bf = cv2.BFMatcher()
all_matches = bf.knnMatch(des1, des2, k=2)
good = []
# give an arbitrary number -> 0.789
# if good -> append to list of good matches
for (m, n) in all_matches:
if m.distance < 0.789 * n.distance:
good.append([m])
if len(good) > max_val:
max_val = len(good)
max_pt = i
max_kp = kp2
print(i, ' ', training_set[i], ' ', len(good))
52



# read image as is
def read_img(file_name):
img = cv2.imread(file_name)
return img


# resize image with fixed aspect ratio
def resize_img(image, scale):
res = cv2.resize(image, None, fx=scale, fy=scale, interpolation =
cv2.INTER_AREA)
return res


# convert image to grayscale
def img_to_gray(image):
img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
return img_gray


# gaussian blurred grayscale
def img_to_gaussian_gray(image):
img_gray = cv2.GaussianBlur(img_to_gray(image), (5, 5), 0)
return img_gray


# convert image to negative
def img_to_neg(image):
utils.py:
if max_val != 8:
print(training_set[max_pt])
print('good matches ', max_val)
train_img = cv2.imread(training_set[max_pt])
img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
note = str(training_set[max_pt])[6:-4]
print('\nDetected denomination: Rs. ', note)
engine = pyttsx3.init()
engine.say(note)
engine.runAndWait()
(plt.imshow(img3), plt.show())
else:
print('No Matches')
engine = pyttsx3.init()
engine.say("this is fake note")
engine.runAndWait()
53
img_neg = 255 - image
return img_neg


# binarize (threshold)
# retval not used currently
def binary_thresh(image, threshold):
retval, img_thresh = cv2.threshold(image, threshold, 255,
cv2.THRESH_BINARY)
return img_thresh


# NO IDEA HOW THIS WPRKS
def adaptive_thresh(image):
img_thresh = cv2.adaptiveThreshold(image, 255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
# cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType,
blockSize, C[, dst]) → dsta
return img_thresh


# sobel edge operator
def sobel_edge(image, align):
img_horiz = cv2.Sobel(image, cv2.CV_8U, 0, 1)
img_vert = cv2.Sobel(image, cv2.CV_8U, 1, 0)
if align == 'h':
return img_horiz
elif align == 'v':
return img_vert
else:
print('use h or v')


# sobel edge x + y
def sobel_edge2(image):
# ksize = size of extended sobel kernel
grad_x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize=3, borderType =
cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize=3, borderType =
cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
dst = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
return dst


# canny edge operator
def canny_edge(image, block_size, ksize):
# block_size => Neighborhood size
# ksize => Aperture parameter for the Sobel operator
# 350, 350 => for smaller 500
# 720, 350 => Devnagari 500, Reserve bank of India
img = cv2.Canny(image, block_size, ksize)
54
# dilate to fill up the numbers
#img = cv2.dilate(img, None)
return img


# laplacian edge
def laplacian_edge(image):
# good for text
img = cv2.Laplacian(image, cv2.CV_8U)
return img


# detect countours
def find_contours(image):
(_, contours, _) = cv2.findContours(image, cv2.RETR_LIST,
cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
return contours


# median blur
def median_blur(image):
blurred_img = cv2.medianBlur(image, 3)
return blurred_img
# dialte image to close lines
def dilate_img(image):
img = cv2.dilate(image, np.ones((5,5), np.uint8))
return img


# erode image
def close(image):
img = cv2.Canny(image, 75, 300)
img = cv2.dilate(img, None)
img = cv2.erode(img, None)
return img

def harris_edge(image):
img_gray = np.float32(image)
corners = cv2.goodFeaturesToTrack(img_gray, 4, 0.03, 200, None, None,
2,useHarrisDetector=True, k=0.04)
corners = np.int0(corners)
for corner in corners:
x, y = corner.ravel()
cv2.circle(image, (x, y), 3, 255, -1)
return image


# calculate histogram
def histogram(image):
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[,
55
accumulate]])
plt.plot(hist)
plt.show()


# fast fourier transform
def fourier(image):
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('FFT'), plt.xticks([]), plt.yticks([])
plt.show()


# calculate scale and fit into display
def display(window_name, image):
screen_res = 1440, 900 # MacBook Air
scale_width = screen_res[0] / image.shape[1]
scale_height = screen_res[1] / image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)
# reescale the resolution of the window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, window_width, window_height)
# display image
cv2.imshow(window_name, image)
# wait for any key to quit the program
cv2.waitKey(0)
cv2.destroyAllWindows()
