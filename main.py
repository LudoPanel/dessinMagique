import cv2, kMean, utils

#######
## Parametres a saisir par l'utilisateur
#######

nomImage = 'caillou.png'
nombreDeCouleur = 6
aireZoneMax =100
largeurZoneMax = 2
rayonDisqueCouleur = 4

#######
## Fin parametre a saisir par l'utilisateur
#######

img = cv2.imread('Images/' + nomImage)
utils.displayImage(img, 'Image initiale')

# On applique kmean pour reduire le nombre de couleurs de l'image
imgKMean = kMean.reducColor(img, nombreDeCouleur)
utils.displayImage(imgKMean, 'Image kmean')

# On nettoie le resultat obtenu (on enleve les zones trop petites)
imgNettoyee = kMean.nettoyageImage(imgKMean, aireZoneMax, largeurZoneMax)
utils.displayImage(imgNettoyee, 'Image nettoyee')

# On extrait les contours de l'image
imgContour = kMean.traceContour(img, imgNettoyee)
utils.displayImage(imgContour, 'Image contour')

# On pose les indications dans les zones
# imgFinale = kMean.ajouterIndicationCouleursZoneLabellisation(imgContour, imgNettoyee, rayonDisqueCouleur)
# utils.displayImage(imgFinale, 'Image finale methode 1')

imgFinale = kMean.ajouterIndicationCouleursZoneErosionSuccessives(imgContour, imgNettoyee, rayonDisqueCouleur)
utils.displayImage(imgFinale, 'Image finale methode 2')