import cv2
import numpy as np


def displayImage(img, titre):
    cv2.imshow(titre, img)
    cv2.waitKey(0)


# Permet d'appliquer un seuil sur une image
# Retour : Image seuillee
def appliquerSeuil(img, seuil):
    index = img[:] > seuil
    img[:] = 255
    img[index] = 0
    return img


# Elargit les trous et retrecit les montagnes (ajoute du fonce et enleve du clair)
# Retour : Image erodee
def erode(img, structElement):
    return cv2.erode(img, structElement)


# Elargit les montagne et retrecit les trous (ajoute du clair et enleve du fonce)
# Retour : Image dilatee
def dilate(img, structElement):
    return cv2.dilate(img, structElement)


# Extraction de contour (dilatation - erosion)
# Retour : Image avec gradient appliquee
def gradient(img, structElement):
    return dilate(img, structElement) - erode(img, structElement)


# Permet de raser les montagnes moins large que structElement
# Retour : Image ouverte
def ouverture(img, structElement):
    return dilate(erode(img, structElement), structElement)


# Permet de combler les ravins plus etroits que structElement
# Retour : Image fermee
def fermeture(img, structElement):
    return erode(dilate(img, structElement), structElement)


# Permet d'extraire le fond d'une image afin de soustraire les effets d'un gradient d'eclairage
# Retour : Image ouverte
def topHat(img, structElement):
    img = img - ouverture(img, structElement)
    return img


def dilateCond(img, marq, el):
    return np.minimum(img, dilate(marq, el))


def erodeCond(img, marq, el):
    return np.maximum(img, erode(marq, el))


def reconstructionInferieure(img, marq, el):
    m1 = dilateCond(img, marq, el)
    while not np.array_equal(marq, m1):
        m1 = marq
        marq = dilateCond(img, m1, el)
    return m1


def reconstructionSuperieure(img, marq, el):
    m1 = erodeCond(img, marq, el)
    while not np.array_equal(marq, m1):
        m1 = marq
        marq = erodeCond(img, m1, el)
    return m1


# Permet de supprimer le bruit "sel" d'une image
# Retour : Image ouverte, puis reconstruction de l'image par son ouvert
def ouvertureReconstruction(img, elementOuverture, elementReconstruction):
    return reconstructionInferieure(img, ouverture(img, elementOuverture), elementReconstruction)


# Permet de supprimer le bruit "poivre" d'une image
# Retour : Image fermee, puis reconstruction de l'image par sa fermeture
def fermetureReconstruction(img, elementFermeture, elementReconstruction):
    return reconstructionSuperieure(img, fermeture(img, elementFermeture), elementReconstruction)
