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

    elementOuverture = strel.build('disque', 1, None)
    elementReconstruction = strel.build('carre', 1, None)

    imgContour = utils.ouvertureReconstruction(imgContour, elementOuverture, elementReconstruction)
    imgContour = utils.fermetureReconstruction(imgContour, elementOuverture, elementReconstruction)

    return imgContour


def nettoyageImage(image, aireZoneMax, largeurMax):

    # On transforme l'image en niveau de gris
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # On transforme l'image en binaire en appliquant un threshold
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    # On recupere les composantes connexes de notre image (technique de labelisation)
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

    # On recupere les labels de nos zones
    labels = output[1]

    # On recupere les statistiques associees a chaque zone
    stats = output[2]

    imgNettoyee = image
    taille = image.shape

    # On itere sur nos pixels
    # Pour chaque pixel, on recupere la zone associee afin de connaitre l'aire et la largeur de la bounding box
    # Si c'est trop petit, alors on remplace la couleur par une couleur precedente
    for x in range(0, taille[0]):
        for y in range(0, taille[1]):
            zone = labels[x][y]
            if stats[zone, cv2.CC_STAT_AREA] < aireZoneMax or stats[zone, cv2.cv2.CC_STAT_WIDTH] < largeurMax:
                imgNettoyee[x, y] = imgNettoyee[x - 2, y]

    return imgNettoyee


def ajouterIndicationCouleursZone(imgContour, imgNettoyee, rayonDisqueCouleur):

    imgray = cv2.cvtColor(imgContour, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    centroids = output[3]

    structElement = strel.build_as_list('disque', rayonDisqueCouleur, None)

    taille = imgContour.shape
    for centroid in centroids[0:output[0]]:
        for (i, j) in structElement:
            if int(centroid[1]) < (taille[1] - rayonDisqueCouleur) and int(centroid[0]) < (taille[0] - rayonDisqueCouleur):
                imgContour[i + int(centroid[1]), j + int(centroid[0])] = imgNettoyee[int(centroid[1]), int(centroid[0])]

    return imgContour