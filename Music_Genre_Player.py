from tkinter import *
import pygame
from pygame import mixer
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tempfile import TemporaryFile
import os
import math
import pickle
import random
import operator


def playsong(self):
    self.track.set(self.playlist.get(ACTIVE))
    self.status.set("-playing")
    pygame.mixer.music.load(self.playlist.get(ACTIVE))
    pygame.mixer.music.play()


def stopsong(self):
    self.status.set("-stopped")
    pygame.mixer.music.stop()
    
    
def pausesong(self):
    self.status.set("-paused")
    pygame.mixer.music.pause()


def unpausesong(self):
    self.status.set("-playing")
    pygame.mixer.music.unpause()


def play_music():
    mixer.music.play()

def pause_music():
    mixer.music.pause()

def stop_song():
    mixer.music.stop()
    
def distance(instance1 , instance2 , k ):
     distance = 0
     mm1 = instance1[0]
     cm1 = instance1[1]
     mm2 = instance2[0]
     cm2 = instance2[1]
     distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
     distance += (np.dot(np.dot((mm2-mm1).transpose(), np.linalg.inv(cm2)), mm2-mm1))
     distance += (np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1)))
     distance -= k
     return distance


def getNeighbors(trainingset, instance, k):
    distances = []
    for x in range(len(trainingset)):
        dist = distance(trainingset[x], instance, k) + distance(instance, trainingset[x], k)
        distances.append((trainingset[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

    
def nearestClass(neighbors):
    classVote = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1
    return 1.0*correct/len(testSet)


class MusicPlayer(object):
    def __init__(self,root):
        self.root = root
        self.root.title("MusicPlayer")
        self.root.geometry("1000x200+200+200")
        pygame.init()
        pygame.mixer.init()
        self.track = StringVar()
        self.status = StringVar()


        trackframe = LabelFrame(self.root, text = "Song Track", font = ("times new roman", 15, "bold"), bg = "blue", fg = "black", bd = 5, relief = GROOVE)
        trackframe.place(x=0, y=0, width=600, height=100)

        songtrack = Label(trackframe, textvariable=self.track, width=20, font=("times new roman", 13, "bold"),bg = "darkblue", fg="white").grid(row=0,column=0,padx=10,pady=5)
        trackstatus = Label(trackframe, textvariable=self.status, font=("times new roman", 10, "bold"),bg = "pink", fg="violet").grid(row=0,column=1,padx=10,pady=5)

        buttonframe = LabelFrame(self.root, text = "Control Panel", font = ("times new roman", 15, "bold"), bg = "orange", fg = "blue", bd = 5, relief = GROOVE)
        buttonframe.place(x=0, y=100, width=600, height=100)

        playbtn = Button(buttonframe, text = "PLAYSONG", command = play_music, width=10, height=1, font=("times new roman", 16, "bold"),bg = "yellow", fg="violet").grid(row=0,column=0,padx=10,pady=5)
        stopbtn = Button(buttonframe, text = "STOPSONG", command = stop_song, width=8, height=1, font=("times new roman", 16, "bold"),bg = "yellow", fg="violet").grid(row=0,column=1,padx=10,pady=5)
        pausebtn = Button(buttonframe, text = "PAUSE", command = pause_music, width=8, height=1, font=("times new roman", 16, "bold"),bg = "yellow", fg="violet").grid(row=0,column=2,padx=10,pady=5)
        unpausebtn = Button(buttonframe, text = "GENRE", command = GenreResult, width=10, height=1, font=("times new roman", 16, "bold"),bg = "yellow", fg="violet").grid(row=0,column=3,padx=10,pady=5)

        songsframe = LabelFrame(self.root, text = "Song Playlist", font = ("times new roman", 15, "bold"), bg ="pink", fg = "white", bd = 5, relief = GROOVE)
        songsframe.place(x=600, y=0, width=400, height=200)

        scrol_y = Scrollbar(songsframe, orient = VERTICAL)
        self.playlist = Listbox(songsframe, yscrollcommand = scrol_y.set, selectbackground = "gold", selectmode = SINGLE, font =("times new roman", 10, "bold"),bg = "pink", fg="violet",relief=GROOVE)
        scrol_y.pack(side = RIGHT, fill = Y)
        scrol_y.config(command=self.playlist.yview)
        self.playlist.pack(fill = BOTH)
        os.chdir(r'C:/Users/sanka/Desktop/New folder/python/Data/Songs')
        songtracks = os.listdir()

        for track in songtracks:
            self.playlist.insert(END, track)
        print(track)


        self.track.set(self.playlist.get(ACTIVE))
        print(self.playlist.get(ACTIVE))
        self.status.set("-playing")
        mixer.music.load(self.playlist.get(ACTIVE))
        mixer.music.pause()
        mixer.music.stop()                                
                                
directory123 = "C:/Users/sanka/Desktop/New folder/python/Data/genres_original"
fS = open("my1.dat", 'wb')
iS = 0
#folderS = " "
for folderS in os.listdir(directory123):
    iS += 1
    if iS == 11:
        break
    for fileS in os.listdir(directory123+"/"+folderS):
       # try:
            (rate, sig) = wav.read(directory123+"/"+folderS+"/"+fileS)
            mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            featureS = (mean_matrix, covariance, iS)
            pickle.dump(featureS, fS)
        #except Exception as e:
           # print("Got an exception: ", e, "in folder: ", folderS, "filename: ", fileS)
fS.close()


dataset = []

def loadDataset(filename, split, trSet, teSet):
    with open("my1.dat", 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


trainingSet = []
testSet = []
loadDataset("my1.dat", 0.66, trainingSet, testSet)


length = len(testSet)
predictions = []
for x in range(length):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))
accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)


from collections import defaultdict
results = defaultdict(int)

directory12 = 'C:/Users/sanka/Desktop/New folder/python/Data/genres_original/blues'
i = 1
for folder in os.listdir(directory12):
    results[i] = folder
    i += 1
print(results[i])

pred = nearestClass(getNeighbors(dataset, featureS, 5))
print(results[pred])
partitioned_string = results[pred].partition('.')
before_first_period = partitioned_string[0]

print(before_first_period)
def GenreResult():
    import ctypes
    from ctypes import wintypes
    ctypes.windll.user32.MessageBoxW(0,(before_first_period), "Genre", 1)


root = Tk()
MusicPlayer(root)
root.mainloop()




        
        
        
    
    
