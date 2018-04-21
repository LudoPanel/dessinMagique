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


def traceContour(imgInitiale, imgNettoyee):

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

    # gestion de la largeur
    elementOuverture = strel.build('carre', largeurMax, None)
    elementReconstruction = strel.build('carre', largeurMax, None)

    imgNettoyee = utils.ouvertureReconstruction(imgNettoyee, elementOuverture, elementReconstruction)
    imgNettoyee = utils.fermetureReconstruction(imgNettoyee, elementOuverture, elementReconstruction)

    # On itere sur nos pixels
    # Pour chaque pixel, on recupere la zone associee afin de connaitre l'aire et la largeur de la bounding box
    # Si c'est trop petit, alors on remplace la couleur par une couleur precedente
    for x in range(0, taille[0]):
        for y in range(0, taille[1]):
            zone = labels[x][y]
            if stats[zone, cv2.CC_STAT_AREA] < aireZoneMax or stats[zone, cv2.cv2.CC_STAT_WIDTH] < largeurMax:
                imgNettoyee[x, y] = imgNettoyee[x - 5, y-5]

    return imgNettoyee


def ajouterIndicationCouleursZoneLabellisation(imgContour, imgNettoyee, rayonDisqueCouleur):

    imgray = cv2.cvtColor(imgContour, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

    # recuperation des composantes connexes
    output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
    centroids = output[3]

    structElement = strel.build_as_list('disque', rayonDisqueCouleur, None)

    # boucle sur les centroids pour placer les cercles de couleurs
    taille = imgContour.shape
    for centroid in centroids[0:output[0]]:
        for (i, j) in structElement:
            if int(centroid[1]) < (taille[1] - rayonDisqueCouleur) and int(centroid[0]) < (taille[0] - rayonDisqueCouleur):
                imgContour[i + int(centroid[1]), j + int(centroid[0])] = imgNettoyee[int(centroid[1]), int(centroid[0])]

    return imgContour


def ajouterIndicationCouleursZoneErosionSuccessives(imgContour, imgNettoyee, rayonDisqueCouleur):

    taille = imgNettoyee.shape
    gamma8 = strel.build('carre', 1, None)

    imgFinale  = np.zeros(taille, np.float)
    imgFinale = np.uint8(imgFinale)

    imgFinale = imgFinale + imgContour

    imgFinale += imgContour

    structElementCercle = strel.build_as_list('disque', rayonDisqueCouleur, None)

    m = np.copy(imgContour)
    imgContour[0, :] = 0
    imgContour[m.shape[0] - 1, :] = 0
    imgContour[:, 0] = 0
    imgContour[:, m.shape[1] - 1] = 0

    # Iteration sur les pixels de imgContour
    for x in range(0, taille[0]):
        for y in range(0, taille[1]):
            if np.any(imgContour[x, y]) != 0:

                # On cherche a isoler la zone
                m[:] = 0
                m[x, y] = 255
                imgReconInf = utils.reconstructionInferieure(imgContour, m, gamma8)

                # Erosion successive jusqu'a faire disparaitre la zone
                continuer = True
                i = 1
                while continuer:
                    structElement = strel.build('disque', i, None)
                    imgErodee = utils.erode(imgReconInf, structElement)

                    # Quand la zone a disparu, alors on prend l'erosion successive precedente afin de placer notre cercle de couleur
                    if np.amax(imgErodee) == 0:
                        structElement = strel.build('disque', i - 1, None)
                        imgErodee = utils.erode(imgReconInf, structElement)

                        # On reboucle afin de chercher un pixel de la zone isolee restant apres les erosions successives
                        for x in range(0, taille[0]):
                            for y in range(0, taille[1]):
                                if np.any(imgErodee[x, y]) != 0:
                                    pixel = (x, y)
                                    break

                        # Boucle permettant de creer notre cercle de couleur a partir de l'element structurant
                        for (i, j) in structElementCercle:
                            if pixel[0] < (taille[1] - rayonDisqueCouleur) and pixel[1] < (taille[0] - rayonDisqueCouleur):
                                imgFinale[pixel[0] + i, j + pixel[1]] = imgNettoyee[pixel[0], pixel[1]]


                        continuer = False

                    i = i + 1

                # on enleve la zone isolee de image contour afin de ne pas traiter plusieurs fois les zones
                imgContour = imgContour - imgReconInf

    return imgFinale