import cv2
import numpy as np
import strel, utils

# Fonction permettant d'appliquer l'algorithme des kMeans, et permet ainsi de reduire le nombre de couleurs de l'image
# Retour : Image avec k couleurs
def reducColor(img, k):

    # Tout d'abord, on redimensionne l'image afin que notre algorithme reste rapide
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)

    # On applique la fonction kmeans sur notre image redimensionnee
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # On redimensionne l'image dans sa taille originale
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


def recupContour(imgInitiale, imgNettoyee):

    # On applique un gradient avec un gamma8
    structElement = strel.build('diamant', 1, 0)

    imageNiveauGris = cv2.cvtColor(imgNettoyee, cv2.COLOR_BGR2GRAY)
    imgAvecGradient = utils.gradient(imageNiveauGris, structElement)

    # Puis on applique un seuil
    seuil = 1
    imgSeuilee = utils.appliquerSeuil(imgAvecGradient, seuil)

    # On reconstitue une image couleur a partir de l'image en niveau de gris
    imgContour = imgInitiale

    index = imgSeuilee[:] == 255
    imgContour[:] = (0, 0, 0)
    imgContour[index] = (255, 255, 255)

    return imgContour


# def nettoyageImage(image, pixelMax, largeurMax):
#
#     # Threshold it so it becomes binary
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
#
#     # You need to choose 4 or 8 for connectivity type
#     connectivity = 4
#
#     # Perform the operation
#     output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
#
#
#
#     return imageNettoye