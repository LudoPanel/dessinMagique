import numpy as np
import cv2, kMean, utils, strel

img = cv2.imread('Images/caillou.png')
utils.displayImage(img, 'Image initiale')

# On applique kmean
imgKMean = kMean.reducColor(img, 8)
utils.displayImage(imgKMean, 'Image kmean')

# On nettoie le resultat obtenu (on enleve les zones trop petites)


# On extrait les contours de l'image
imgContour = kMean.recupContour(img, imgKMean)
utils.displayImage(imgContour, 'Image contour')

# On pose les indications dans les zones



# Affichage du resultat final
