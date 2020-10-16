import sys
import getopt
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import functools
from createTeeth2Dataset import isTeethColor
# const
I = None
I_dummy = None
l_range = [0, 256]
a_range = [0, 256]
b_range = [0, 256]
lab_bins = [32, 32, 32]

# class
class Superpixel():
    def __init__(self):
        self.name = ''
        self.pixels = []
        self.id = None

    def setID(self, id):
        self.id = id

    def getID(self, id):
        return self.id

    def addPixel(self, pixel):
        self.pixels.append(pixel)

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name
    def getMeanDensity(self): 
        return [int(round(sum(pixel)/len(self.pixels))) for pixel in zip(*self.pixels)]

    def getNumberOfPixels(self):
        return len(self.pixels)





def genSuperpixelSEED(image):
    numberOfSuperpixel = 500
    iterator = 100
    blockLevels = 10
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        image.shape[1], image.shape[0], image.shape[2], numberOfSuperpixel, blockLevels, prior=2, histogram_bins=5, double_step=False)
    seeds.iterate(image, num_iterations=iterator)
    return seeds


def getSuperpixels(image, labels, numberOfSuperpixelResult):
    superpixels = []
    for i in range(0, numberOfSuperpixelResult):
        newSuperpixel = Superpixel()
        newSuperpixel.setID(i)
        superpixels.append(newSuperpixel)
    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            superpixels[labels[i][j]].addPixel(image[i][j])
    return superpixels


def genSuperpixelSLIC(image, regionSize):
    SLIC = 100
    SLICO = 101
    num_iter = 100
    slics = cv2.ximgproc.createSuperpixelSLIC(
        image, region_size=regionSize, ruler=20.0)
    slics.iterate(num_iterations=num_iter)

    return slics


def genSuperpixelLSC(image, regionSize):
    SLIC = 100
    SLICO = 101
    num_iter = 100
    slcs = cv2.ximgproc.createSuperpixelLSC(image, region_size=regionSize)
    slcs.iterate(num_iter)
    return slcs


def preprocessing(image, seeds, teethColor):
    numberOfSuperpixelResult = seeds.getNumberOfSuperpixels()
    labels = seeds.getLabels()
    superpixels = getSuperpixels(image, labels, numberOfSuperpixelResult)
    
    # 1M2: RGB 221, 212, 180
    
    isTeeth = 0
    for superpixel in superpixels:
        density = superpixel.getMeanDensity()
        if isTeethColor([density[2], density[1], density[0]], teethColor, 20):
            superpixel.setName("teeth")
            isTeeth = isTeeth + 1
        else:
            superpixel.setName("notTeeth")
    print(f"isTeeth: {isTeeth}")

    for i in range(0, labels.shape[0]):
        for j in range(0, labels.shape[1]):
            if superpixels[labels[i][j]].getName() == "notTeeth":
                image[i][j] = [teethColor[2], teethColor[1], teethColor[0]]
    
    return image




if __name__ == "__main__":
    image = cv2.imread("images/braces.png")
    imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    teethColor = [221, 212, 180]
    seeds = genSuperpixelSEED(image)
    slics = genSuperpixelSLIC(image, 10)
    lscs = genSuperpixelLSC(image, 10)
    print(image.shape)
    cv2.imshow("a", preprocessing(image, seeds))
    cv2.waitKey()