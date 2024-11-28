# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:16:10 2023

@author: Tastat
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from matplotlib.colors import LogNorm
import scipy.special

# =============================================================================
# Aufgabe 1.1 - Quellenbild erstellen
# =============================================================================

QA = np.zeros((110,110), dtype=int)
QA [0:100,10:110] = np.random.poisson(100,(100,100))

#generates full array with value 50, t = boolean mask that generates the lines 
#if multiplied with a value, last step generates the noise and dark lines
QB = np.full((100,100),50,dtype=int)
t = np.linspace(start=np.zeros(100),stop=99,num=100,axis=1)%10
QB = QB * np.where(t>4,6*np.random.poisson(300,(100,100))/300,
                   np.random.poisson(50,(100,100))/50)

#für den Quadranten C wäre der Mittelpunkt bei QC[40,40]
QC = np.zeros((110,110),dtype=int)

#bei den Koordinaten beachte, dass Arrays bei 0 Anfangen (M(50,50) -> M(49,49))
MZ = 59 #gibt die Lage des Mittelpunkts vom Kreis auf der y-Achse (Zeilen) an
MS = 50 #gibt die Lage des Mittelpunkts vom Kreis auf der x-Achse (Spalten) an
#dont question the MS=50 it just is 50 (controlled by counting the sides)
# QC[60,50] = 1000 #sollte den Mittelpunkt markieren

for i in range(0,len(QC)):  #Spalten
    for j in range(0,int(np.size(QC)/len(QC))):   #Zeilen
        if np.sqrt((MZ-i)**2+(MS-j)**2) <= 50.5:    #-i=Zeile -j=Spalte,
            QC[i,j] = np.random.poisson(150) 
            
QD = np.zeros((127,127))
# QD[59,9] = 2000

#Die Schleife soll alle Zellen markieren die zwischen y1 und y2 sind.Die
#Funktion y1 soll die obere y2 die untere Seite des Dreiecks sein.
for i in range(10,110):                             #Spalten
    for j in range(0,int(np.size(QD)/len(QD))):     #Zeilen
       if j <= 58.5/100 * (i-10) + 59:  #Gerade für untere Begrenzung
           if j >= -58.5/100 * (i-10) + 59: #Gerade für obere Begrenzung
               QD[j,i] = np.random.poisson(200)

Bild = np.zeros((256,256))
Bild[17:127,128:238] = QA
Bild[17:117,17:117] = QB   #andere Koordinaten, weil das Array (100,100) ist
Bild[128:238,17:127] = QC
Bild[128:255,128:255] = QD #andere Koordinaten, weil das Array (127,127) ist

#normiert die Grauwerte auf 0 bis 255
Bild = np.round(Bild/np.max(Bild)*255)

plt.figure(1)
plt.imshow(Bild,cmap='gray',extent=[-128, 127, -128, 127])
plt.title('Ausgangsbild')
plt.show()

# =============================================================================
# Plotting
# =============================================================================

def Plot(Bild,title,subplot=None,xlabel='x-Achse',ylabel='y-Achse',legend='n'):
    """

    Parameters
    ----------
    Bild : Numpy.Array as float/int
        Eingangsbild als Numpy.Array
    title : str
        Titel den der Plot haben soll.
    subplot : int as tuple, optional
        gibt an ob Subplots erstellt werden sollen und in welcher Dimension.
        The default is None.
    xlabel : str, optional
        Beschriftung der x-Achse vom Plot. The default is 'x-Achse'.
    ylabel : TYPE, optional
        Beschriftung der y-Achse vom Plot. The default is 'y-Achse'.
    legend : str, optional
        Wird für die Abfrage verwendet ob eine Legende eingefügt werden soll.
        Andere Datentypen sind auch möglich. The default is 'n'.

    Returns
    -------
    None.

    """
    plt.figure()
    
    if subplot != None:
        plt.subplot(subplot)
        
    else:
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.imshow(Bild,cmap='gray',extent=[-128, 127, -128, 127])
        if legend != 'n':
           plt.legend()
        
    plt.show()


# =============================================================================
# Aufgabe 2.1 - Grauwertprofile entlang y = 60 und y = -60
# =============================================================================

#zur Orientierung [:,127] = -1 -> -60 = [:,68] und [:,128] = 1 -> +60 = [:,187]
def Grauwertprofil(Richtung,a,Bild):
    """
    Erstellt ein Grauwertprofil entlang der x-Achse, wenn Richtung = 'x' oder
    entlang der y-Achse, wenn Richtung = 'y'

    Parameters
    ----------
    Richtung: str
        gibt an entlang welcher Achse das Grauwertprofil erstellt werden soll.
    a : int
        x/y-Koordinate entlang welcher das Grauwertprofil erstellt werden soll.
    Bild : numpy.array
        Bild als Array von welchem das Grauwertprofil erstellt werden soll.

    Returns
    -------
    line: numpy.array
        die Grauwerte des Bildes für die geforderten Koordinaten in einem Array
        der Größe (1,len(Bild))

    """
    a1 = abs(a - 128)
        
    if Richtung == 'y':
        line = Bild[a1,:]
        
    if Richtung == 'x':
        line = Bild[:,a1]
    
    x = np.linspace(1,len(Bild),len(Bild))
    
    plt.figure()
    plt.plot(x,line,color='blue',label = f'{Richtung} = {a}')
    plt.title(f'Grauwerprofil für {Richtung} = {a}')
    plt.xlabel('x-Achse des Bildes [Pixel]')
    plt.ylabel('Grauwerte')
    plt.legend()
    plt.show()
    
    return line

l1 = Grauwertprofil('y',60, Bild)
l2 = Grauwertprofil('y',-60,Bild)

x = np.linspace(1,len(Bild),len(Bild))
plt.figure()
plt.plot(x,l1,color='blue',label = 'y = 60')
plt.plot(x,l2,color='red',label= 'y = -60')
plt.title('Grauwertprofile zusammengefasst')
plt.xlabel('x-Achse')
plt.ylabel('Grauwerte')
plt.show()

# =============================================================================
# Aufgabe 2.2 - Histogramm
# =============================================================================

def Histogramm(Bild,b=256,scale=True,titel='Histogramm aller Grauwerter'):
    """
    erstellt ein Histogramm aller Grauwerte von einem Bild

    Parameters
    ----------
    Bild : numpy.array
        Ausgangsmatrix.
    b : int, optional
        Anzahl der bins, standard = 256.
    scale : True/False  optional
        False = lineare y-Achse, standard = True = logarithmische y-Achse.
    titel : str
        Überschrift des Plots 
    Returns
    -------
    None.

    """
    l = np.size(Bild)
    Bild = np.reshape(Bild,newshape=(l,1))
    
    plt.figure()
    plt.hist(Bild,bins=b,log=scale)
    plt.title(titel)
    plt.xlabel('Grauwerte')
    plt.ylabel('Anzahl')
    plt.show()

hist1 = Histogramm(Bild)

# =============================================================================
# Aufgabe 2.3 - Mittelwert und Schiefe Grauwert-Histogramm
# =============================================================================

def MeanAndSkewdness(Bild,returns=()):
    """
    Die Methode berechnet das 2 - 4 Moment eines Bildes (Var, Schiefe, Exzess).
    Die Methode wird für die Methode Informationsgehalt benötigt.
    Die Methode wird wie folgt aufgerufen: 
    MeanAndSkewdness(Bild,returns=('hf','Hf','M2'))
    je nachdem welche Parameter abgerufen werden sollen sind diese in returns
    als str einzutragen

    Parameters
    ----------
    Bild : numpy.array
        Ausgangsmatrix.

    Return options -> enter as str when calling methode
    -------
    f : array of float
        unique (Grau-)Werte im Array.
    Hf : array of int
        Anzahl wie oft ein unique Wert im Array vorkommt.
    hf : array of float
        Prozentualle Anteil den ein f in der Gesamtheit hat (Hf / Arraygröße).
    mean : float
        Mittelwert der Grauwerte vom Array (Sum(f*hf)).
    M2 : float
        zentrales Moment 2-ter Ordnung bzw. Varianz der Grauwerte vom Bild.
    M3 : float
        zentrales Moment 3-ter Ordnung.
    M4 : float
        zentrales Moment 4-ter Ordnung.
    skew : float
        Schiefe des Bildes aus 2-ter und 3-ter Ordnung (M3/M2**(1.5)).

    Return
    -------
    returns : tuple (array of float + array of int + float)
        enthält die Ausgabe der gewünschten Parameter aus Return options       

    """

    #erstellt 2 Arrays (f) einmal mit den unique Values und (Hf) Anzahl des
    #jeweiligen unique Values
    f, Hf = np.unique(Bild,return_counts=True)
    hf = Hf / np.size(Bild)

    mean = np.sum(hf*f) #np.mean(Bild) macht es auch... 
    
    #zentrale Momente n-ter Ordnung
    M2 = np.sum((f-mean)**2*hf) #Varianz
    M3 = np.sum((f-mean)**3*hf) #3.tes Moment Schiefe
   
    results = []
    
    for value in returns:
        if value == 'f':
            results.append(f)
        elif value == 'Hf':
            results.append(Hf)
        elif value == 'hf':
            results.append(hf)
        elif value == 'mean':
            results.append(mean)
        elif value == 'M2':
            results.append(M2)
        elif value == 'M3':
            results.append(M3)
        elif value == 'M4':
            results.append(np.sum((f-mean)**4*hf))
        elif value == 'skew':
           results.append(M3 / M2**(3/2))       
        
    return tuple(results)

hf1,mean1,skew1 = MeanAndSkewdness(Bild,returns=('hf','mean','skew'))


# =============================================================================
# Aufgabe 2.4 - mittlerer Informationsgehalt
# =============================================================================

def Informationsgehalt(hf):
    """
    Die Methode gibt den mittleren Informationsgehalt eines Bildes wider. Für
    die Methode wird vorher hf aus der Methode MeanAndSkewdness benötigt.

    Parameters
    ----------
    hf : numpy.array of float
        Prozentualer Anteil eines unique Grauwerts von allen Werten im Array.
        Siehe Methode MeanAndSkewdness.

    Returns
    -------
    I : float
        mittlerer Informationsgehalt eines Bildes. Je größer desto besser.
        Maximum liegt bei der Bittiefe des Bildes.

    """
    I  = np.sum((-hf * np.log2(hf)))
    
    return I

I1 = Informationsgehalt(hf1)


# =============================================================================
# Aufgabe 2.5 - Bitebenen
# =============================================================================

def Bitebenen(Bild):
    
    n = math.ceil(np.log2(np.max(Bild)))    #Anzahl der Bitebenen
    I = []
        
    bitebene = np.where(Bild-2**(n-1)>=0,1,0)
    laufwerte = Bild - bitebene * 2**(n-1)
    hf,f = MeanAndSkewdness(bitebene, returns=('hf','f'))
    I = np.append(I,(Informationsgehalt(hf)))
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,4,1)
    plt.imshow(bitebene,cmap='gray')
    plt.title(f'Bitebene {n-1}')
    plt.axis('off')
    
    for i in range(n-2,-1,-1):
        bitebene = np.where(laufwerte-2**i>=0,1,0)
        laufwerte = laufwerte - bitebene * 2**i
        
        hf,f = MeanAndSkewdness(bitebene, returns=('hf','f'))
        I = np.append(I,(Informationsgehalt(hf)))
        
        plt.subplot(2, 4, 8 - i)
        plt.imshow(bitebene, cmap='gray')
        plt.title(f'Bitebene {i}')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show
    
    return I
    
Ibis = Bitebenen(Bild)


# =============================================================================
# Aufgabe 2.6 - Differenzbild
# =============================================================================        
    
def Differenzbild(Bild):
    """
    Berechnet das Differenzbild. Das Differenzbild ist die Differenz vom Pixel
    zum vorherigen Pixel in der selben Zeile (Spalte 0 = Ausgangswerte)

    Parameters
    ----------
    Bild : numpy.array of float
        Ausgangsmatrix.

    Returns
    -------
    diff : numpy.array of float
        Array des Differenzbildes.

    """
    
    w = int(np.size(Bild)/len(Bild))    #Breite des Bildes
    
    diff = np.zeros((int(len(Bild)),w)) #leere Matrix
    diff[:,0] = Bild[:,0]               #füllt erste Spalte mit Ausgangswerten
    
    for i in range(1,w):    #berechnet Spaltenweise Differenz
        
        diff[:,i] = Bild[:,i] - Bild[:,i-1]
        
    c = np.max(abs(diff[:,1:])) #Berechnet Summand um keine negativen 
    diff[:,1:] = diff[:,1:] + c #HUs zu erhalten
    diff = np.round(diff/np.max(diff)*255)
    
    Plot(diff,'Differenzbild')
    return diff,c

dBild,c = Differenzbild(Bild)

Histogramm(dBild,titel='Histogramm des Differenzbilds')
hf2,mean2 = MeanAndSkewdness(dBild,returns=('hf','mean'))
Idiff = Informationsgehalt(hf2)
#der mittlere Informationsgehalt ist im vergleich zum Originalbild verringert

# =============================================================================
# Aufgabe 2.7 - Fouriertransformation    
# =============================================================================
    
def Fouriertransformation(Bild,Log=True):
    
    fft = np.fft.fft2(Bild)
    fft = np.fft.fftshift(fft)
    fft = abs(fft)**2
    
    if Log == True:
        scale = LogNorm()
    else:
        scale = scale = plt.Normalize(vmin=fft.min(),vmax=fft.max())
    
    Plot(fft,'Powerspektrum Funktion')
    plt.figure()
    plt.imshow(fft,cmap='gray',norm=scale)
    plt.title('Powerspektrum')
    plt.show()
    return fft

fft = Fouriertransformation(Bild)


# =============================================================================
# Aufgabe 2.8 - Drehen um 30°
# =============================================================================

def Rotation(Bild,Winkel):
    
    r = rotate(Bild,angle=Winkel,reshape=False)
    Plot(r, 'rotiertes Bild')

    return r

r = Rotation(Bild,30)
Fouriertransformation(r)

# =============================================================================
# Aufgabe 2.9 - Tiefpassfilter
# =============================================================================

def Tiefpassfilter(Bild,Frequenz):
    """
    Das Bild muss hierfür quadratisch sein (ich kann das abändern indem ich zwei linspaces generiere)

    Parameters
    ----------
    Bild : TYPE
        DESCRIPTION.
    Frequenz : float
        Grenzfrequenz für den Filter in vielfachen der Nyquist-Frequenz.

    Returns
    -------
    None.

    """
 
    a = np.linspace(-0.5,0.5,len(Bild))
    x,y = np.meshgrid(a,a)  #berechnet den Betrag vom Abtand zum Mittelpunkt
    xy = np.sqrt(x**2+y**2) #alle Felder über 0.25 (=0.5 nyquist frequenz) 
                            #werden abgeschnitten
                            #1/4 niquist frequenz = 0.5 * 0.25 = 0.125
    tiefpass = np.where(xy>0.5 * Frequenz,0,1)   
    
    fft = np.fft.fft2(Bild)
    fft = np.fft.fftshift(fft)
    
    gefiltert = tiefpass * fft
    
    ass = abs(np.fft.ifft2(gefiltert))
    
    Plot(ass, 'Bild gefaltet mit Tiefpassfilter')

Tiefpassfilter(Bild, 0.25)
    
# =============================================================================
# Aufgabe 2.10 - Bandpassfilter
# =============================================================================

def Bandpassfilter(Bild,Frequenz1,Frequenz2): #vgl Tiefpassfilter
    
    a = np.linspace(-0.5,0.5,len(Bild))
    x,y = np.meshgrid(a,a)
    xy = np.sqrt(x**2+y**2)
    bandpass = np.where((xy>=0.5*Frequenz1)&(xy<=0.5*Frequenz2),1,0)
    
    fft = np.fft.fft2(Bild)
    fft = np.fft.fftshift(fft)
    
    gefiltert = bandpass * fft
    
    ass = abs(np.fft.ifft2(gefiltert))
    
    Plot(ass,'Bandpassfilter')
    
Bandpassfilter(Bild,0.375,0.625)

# =============================================================================
# Aufgabe 2.11 - Hochpassfilter
# =============================================================================

def Hochpassfilter(Bild,Frequenz):  #vgl Tiefpassfilter
       
    a = np.linspace(-0.5,0.5,len(Bild))
    x,y = np.meshgrid(a,a)  #berechnet den Betrag vom Abtand zum Mittelpunkt
    xy = np.sqrt(x**2+y**2)
    tiefpass = np.where(xy<=0.5 * Frequenz,0,1)
    
    fft = np.fft.fft2(Bild)
    fft = np.fft.fftshift(fft)
    
    gefiltert = tiefpass * fft
    
    ass = abs(np.fft.ifft2(gefiltert))
    
    Plot(ass,'Hochpassfilter')


Hochpassfilter(Bild,0.75)


# =============================================================================
# Aufgabe 3.1 - Graukeil und Kennlinien
# =============================================================================

#lineare Kennlinie = Grauwertkeil
#Grauwertkeil mit gleicher größe wie Bild damit übersichtlicher
Keil = np.tile(np.linspace(0,255,256),(256,1))  

#negative Kennlinie
Negativ = 255-Keil

#quadratische Kennlinie
Quadrat = np.floor(Keil**2/255)

#Wurzel-Kennlinie
Wurzel = np.floor(np.sqrt(255*Keil))

#binarisiert
fu = 128-64
fo = 128+64
#fu = int(input('untere Grenze Grauwerte: ')) #alternativ Grenze per Input 
#fo = int(input('obere Grenze Grauwerte: '))  #festlegen lassen
Binar = np.where(Keil<fu,0,1) * np.where(Keil>fo,0,1)

#gaussian
Gauß = np.floor(255/2 * (1+scipy.special.erf((Keil-128)/(np.sqrt(2)*40))))

plt.Figure()
plt.subplot(231)
plt.imshow(Keil,cmap='gray')
plt.title('lineare Kennlinie')
plt.subplot(232)
plt.imshow(Negativ,cmap='gray')
plt.title('Negative Kennlinie')
plt.subplot(233)
plt.imshow(Quadrat,cmap='gray')
plt.title('quadratische Kennlinie')
plt.subplot(234)
plt.imshow(Wurzel,cmap='gray')
plt.title('Wurzel-Kennline')
plt.subplot(235)
plt.imshow(Binar,cmap='gray')
plt.title('binäre Kennlinie')
plt.subplot(236)
plt.imshow(Gauß,cmap='gray')
plt.title('Gauß-Kennlinie')
plt.tight_layout()
plt.show()


# =============================================================================
# Aufgabe 3.2- Rotation und Scherung
# =============================================================================

def Scherung(Bild,ax,ay):
        
    scherungsmatrix = np.array([[1,ax],[0,ay]])
    Bild2 = np.zeros((256,256))
    width = int(np.size(Bild)/len(Bild))
    
    for i in range(0,len(Bild)):
        for j in range(0,width):
            x = j-128
            y = 128-i
            a = np.array([x,y])
            b = Bild[i,j]
            
            # at = np.dot(scherungsmatrix,a)
            at = scherungsmatrix@a
            ax = int(128+round(at[0]))
            ay = int(128-round(at[1]))
            
            if ax >= width or ay >= len(Bild):
                continue
            elif ax < 0:
                continue
            else:
                Bild2[ay,ax] = b
    
    # from scipy.ndimage import affine_transform 
    #-> Scherung verläuft entlang der falschen Axis?
        
    # affine_transform(Bild90,scherungsmatrix)    
    #numpy.pad(Bild,12,mode='constant',constant_value=0.0) #auch nicht korrekt
    Plot(Bild2,'geschertes Bild')

    return Bild2
    
    
    
Bild90 = Rotation(Bild, 90) #rotation im mathemtischem Sinne
a = 1/np.sqrt(2)
BildSchere = Scherung(Bild90.astype(int),a,a)


# =============================================================================
# Aufgabe 3.3 - Mittelwert-, Median- & Binomialfilter
# =============================================================================

def Filtern(Bild,Filter=None):
    """

    Parameters
    ----------
    Bild : numpy.array as float
        Eingangsarray welches gefiltert werden soll.
    Filter : int, optional
        DESCRIPTION. The default is None = alle Filter. 1 = Mittelwertsfilter,
        2= Medianfilter, 3 = Binomialfilter

    Returns
    -------
    Bild1 : TYPE
        DESCRIPTION.
    Bild2 : TYPE
        DESCRIPTION.
    Bild3 : TYPE
        DESCRIPTION.

    """
    
    l = int(len(Bild))
    w = int(np.size(Bild)/len(Bild))
    returns=[]
    
    #Mittelwertfilter
    if Filter == None or '1':
        Bild1 = np.zeros((256,256))
        for i in range(1,l-1):  #Zeilen / y-Richtung
            for j in range(1,w-1):  #Spalten / x-Richtung
                filt = Bild[i-1:i+2,j-1:j+2]
                v = np.mean(filt)
                Bild1[i,j] = v
        returns.append(Bild1)

    #Medianfilter
    if Filter == None or '2':
        Bild2 = np.zeros((256,256))
        for i in range(1,l-1):
            for j in range(1,w-1):
                filt = Bild[i-1:i+2,j-1:j+2]
                v = np.sort(filt.flatten()).tolist()
                Bild2[i,j] = v[4]
        returns.append(Bild2)
        
    #Binomialfilter (Gaußfilter)
    if Filter == None or '3':
        Bild3 = np.zeros((256,256))
        for i in range(1,l-1):
            for j in range(1,w-1):
                filt = Bild[i-1:i+2,j-1:j+2]
                fit = np.array([[1,2,1],[2,4,2],[1,2,1]])
                v = 1/16*fit*filt
                Bild3[i,j] = v.sum()
        returns.append(Bild3)
        
    plt.figure()
    plt.title('Original')
    plt.imshow(Bild,cmap='gray')
    plt.show()
    plt.figure()
    plt.title('Mittelwertsfilter')
    plt.imshow(Bild1,cmap='gray')
    plt.show()
    plt.figure()
    plt.title('Medianfilter')
    plt.imshow(Bild2,cmap='gray')
    plt.show()
    plt.figure()
    plt.title('Binomialfilter')
    plt.imshow(Bild3,cmap='gray')
    plt.show()
            
    return returns

Bild_mit, Bild_med, Bild_bi = Filtern(Bild)
#TODO gibt es unterschiede im Ergebnis? An geeigneten Profilen darstellen

# =============================================================================
# Aufgabe 3.4 Sobel- und Roberts-Filter
# =============================================================================

def Sobelfilter(Bild):
    
    l = int(len(Bild))
    w = int(np.size(Bild)/l)
    
    fft = np.fft.fft2(Bild)
    fft = np.fft.fftshift(fft)
    
    fftx = np.zeros((256,256))
    ffty = np.zeros((256,256))
    
    x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    for i in range(1,l-1):      #Zeilen
        for j in range(1,w-1):  #Spalten
            
            fx = Bild[i-1:i+2,j-1:j+2]
            
            fftx[i,j] = np.sum(np.dot(x,fx))

            ffty[i,j] = np.sum(np.dot(fx,y))   
            
    ffty = abs(ffty)/np.max(abs(ffty))*255
    fftx = abs(fftx)/np.max(abs(fftx))*255
    SobelBild = abs(fftx) + abs(ffty)
    
    SobelBild = abs(SobelBild)/np.max(SobelBild)*255
    
    Plot(SobelBild.astype(int),'Sobel-Filter Funktion')
    
    plt.figure()
    plt.title('SobelFilter')
    plt.imshow(SobelBild.astype(int),cmap='gray')
    plt.show()
    
    return SobelBild
    
Sobelfilter(Bild)


def RobertsFilter(Bild):
    
    l = int(len(Bild))
    w = int(np.size(Bild)/l)
    
    fft = np.fft.fft2(Bild)
    fft = np.fft.fftshift(fft)
    
    fftx = np.zeros((256,256))
    ffty = np.zeros((256,256))
    
    x = np.array([[1,0],[0,-1]])
    y = np.array([[0,1],[-1,0]])
    
    for i in range(1,l-1):      #Zeilen
        for j in range(1,w-1):  #Spalten
            
            fx = Bild[i-1:i+1,j-1:j+1]
            
            fftx[i,j] = np.sum(np.dot(x,fx))

            ffty[i,j] = np.sum(np.dot(fx,y))   
            
    ffty = abs(ffty)/np.max(abs(ffty))*255
    fftx = abs(fftx)/np.max(abs(fftx))*255
    SobelBild = abs(fftx) + abs(ffty)
    
    SobelBild = abs(SobelBild)/np.max(SobelBild)*255
    
    Plot(SobelBild.astype(int),'Roberts Filter Funktion')
    
    plt.figure()
    plt.title('RobertsFilter')
    plt.imshow(SobelBild.astype(int),cmap='gray')
    plt.show()
    
RobertsFilter(Bild)
#TODO unterschiede im erzeugten Bild vom Roberts und Sobel-Filter
#beides extrahiert Kanten, aber entlang unterschiedlicher Raumrichtungen, was
#zu unterschiedlichen Intensitäten der extrahierten Kanten führt



# =============================================================================
# Aufgabe 3.5 - Laplace-Nachbarschaft
# =============================================================================


def Laplace(Bild):
    
    l = int(len(Bild))
    w = int(np.size(Bild)/l)
    
    fftx = np.zeros((256,256))
    ffty = np.zeros((256,256))
    
    x = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    
    for i in range(1,l-1):      #Zeilen
        for j in range(1,w-1):  #Spalten
            
            fx = Bild[i-1:i+2,j-1:j+2]
            
            fftx[i,j] = np.sum(np.dot(fx,x))

    fftx = abs(fftx)/np.max(abs(fftx))*255
    
    Plot(fftx,'Laplace-Filter Funktion')
    plt.figure()
    plt.title('Laplace')
    plt.imshow(fftx,cmap='gray')
    plt.show()
    
    return fftx

Lp = Laplace(Bild)



# =============================================================================
# Aufgabe 3.6 - Schwellwertverfahren
# =============================================================================

#unter der Annahme, dass Flächen aus Aufgabe 1.1 bekannt sind:
def Schwellwertverfahren(Bild):
    
    fb = 5000   #Fläche von Bildobjekt B
    fd = np.sum(QD>0)   #Fläche von Bildobjekt D
    
    MN = 256**2
    hf,f = MeanAndSkewdness(Bild,returns=('hf','f'))
    
    T = 0
    i = np.min(Bild)
    
    while T <= 1-(fb+fd)/MN:
        
        T = np.sum(hf*(f<=i))
        i += 1

    Segment = np.where(Bild>=i,1,0)
    
    T = 0
    i = np.min(Bild)
    
    while T <= 1-fd/MN:
        
        T = np.sum(hf*(f<=i))
        i += 1
        
    Segment2 = np.where(Bild<=i,1,0)
    Segment3 = Segment2 * Segment
    
    Plot(Segment3,'Segmentierung Fläche D ohne Median Filter Funktion')

    return Segment3

Schwellwertverfahren(Bild)

def SchwellwertmitMedian(Bild):
    
    B1,B2,B3 = Filtern(Bild,'Median')
    S1 = Schwellwertverfahren(B2)
    
    return S1
    
    

# =============================================================================
# Aufgabe 3.8 - geometrischen Schwerpunkt und Massenschwerpunkt Fläche A und D
# =============================================================================

def Schwerpunkte(Bild):
    
    BildAD = Bild
    BildAD[17:117,17:117] = 0    #entfernt Flächenquelle B aus Gesamtbild
    BildAD[128:238,17:127] = 0   #entfernt Flächenquelle C aus Gesamtbild
    
    
    #geometrischer Schwerpunkt des Bildes
    #Binarisierung damit Schwerpunkt unabhängig der Grauwerte gebildet wird
    BildADBinär = np.where(BildAD>0,1,0)  
    
    m00 = np.sum(BildADBinär) #Berechnet Fläche der Quellen
    
    m01 =np.sum(BildADBinär,axis=0)#Summiert alle Werte entlang der Y-Achse auf
    m10 =np.sum(BildADBinär,axis=1)#Summiert alle Werte entlang der X-Achse auf
    
    xg = np.sum(m01 * np.linspace(1,256,256))/m00  #Schwerpunkt in x-Richtung
    yg = np.sum(m10 * np.linspace(1,256,256))/m00  #Schwerpunkt in y-Richtung
                                          
    plt.title("geometrischer Schwerpunkt der Flächenquellen A & D")
    plt.imshow(BildAD,cmap='gray')
    plt.scatter(np.round(xg),np.round(yg),marker="+",color='navy')
    plt.show()
    
    xg = xg - 128   #Umrechnung auf Koordinatensystem von Aufgabe 1.1
    yg = 128 - yg   #Umrechnung auf Koordinatensystem von Aufgabe 1.1
    
    #Massenschwerpunkt des Bildes
    m00 = np.sum(BildAD) #Berechnet Summe aller Grauwerte im Bild
    
    m01 = np.sum(BildAD,axis=0)#Summiert alle Grauwerte entlang der Y-Achse auf
    m10 = np.sum(BildAD,axis=1)#Summiert alle Grauwerte entlang der X-Achse auf
    
    xs = np.sum(m01 * np.linspace(1,256,256))/m00  #Schwerpunkt in x-Richtung
    ys = np.sum(m10 * np.linspace(1,256,256))/m00  #Schwerpunkt in y-Richtung 
                                          
    plt.title("Massenschwerpunkt der Flächenquellen A & D")
    plt.imshow(BildAD,cmap='gray')
    plt.scatter(np.round(xs),np.round(ys),marker="+",color='navy')
    plt.show()
    
    xs = xs - 128 #Umrechnung auf Koordinatensystem von Aufgabe 1.1
    ys = 128 - ys #Umrechnung auf Koordinatensystem von Aufgabe 1.1
    
    return xg,yg,xs,ys



# =============================================================================
# Aufgabe 3.9 - Textur und Grauwert-Übergangsmatrizen
# =============================================================================

#isoliert Quadrant B astype(int), weil die Werte für Array-Indizes verwendet
#werden müssen diese den Datentype int besitzen
BildB = np.zeros((128,128))         
BildB = Bild[:128,:128].astype(int) 


l = len(BildB)
w = int(np.size(BildB)/l)   #float geht nicht für: for loop j in range()

cx = np.zeros((256,256))
cy = np.zeros((256,256))

#Übergangsmatrix für Delta=(1,0) (in positive x-Richtung)
for i in range(w-1):
    for j in range(l):
        y = BildB[j,i]      #Ausgangsgrauwert
        x = BildB[j,i+1]    #Übergangswert
        
        cx[y,x] += 1        #Erhöht Übergangsmatrix an entsprechendem Übergang
        
#Übergangsmatrix für Delta=(0,1) (in positive y-Richtung)
for i in range(w):
    for j in range(l-1):
        #Da Übergänge von unten nach oben gezählt werden sollen und j beim
        #Array von oben Anfängt wird hier +1 gerechnet
        y = BildB[j+1,i]    #Ausgangsgrauwert
        x = BildB[j,i]      #Übergangswert
        
        cy[y,x] += 1        #Erhöht Übergangsmatrix an entsprechendem Übergang

plt.title("Übergangsmatrix für \u03B4=(1,0)") #also in x-Richtung
plt.imshow(cx,cmap='hot',vmin=0,vmax=1)
plt.show()
        
plt.title("Übergangsmatrix für \u03B4=(0,1)") #also in y-Richtung
plt.imshow(cy,vmin=0,vmax=1, cmap='hot')
plt.show()

GSx, GSy, MSx, MSy = Schwerpunkte(Bild)
