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


# Permet de supprimer les montagnes (ou des vallees) dont la hauteur (ou profondeur) sont superieur a la hauteur passee en param
def hExtrema(img, structElement, hauteur):
    img[img < hauteur] = hauteur
    imgMoinsConstante = img - 50
    imgReconstruie = reconstructionInferieure(img, imgMoinsConstante, structElement)
    return imgReconstruie


# Permet de mettre en valeur des objets selon leur luminosite locale (exemple : les etoiles)
def extremaRegionaux(img, structElement, hauteur):
    imgHExtrema = hExtrema(img, structElement, hauteur)
    imgExtremas = img - imgHExtrema
    return imgExtremas


# Permet de recuperer le seuil le plus optimise
def seuilOtsu(img):

    # Initialisation
    img = img[:, :, 0]
    histo = cv2.calcHist([img], [0], None, [256], [0, 256])

    taille = img.shape
    N = taille[0] * taille[1]

    mu = np.average(img[:, :])

    h0 = np.sum(histo[0])

    g1 = h0
    mu1 = 0

    g2 = N - h0
    mu2 = (mu*N) / np.abs(g2)

    bestVar = (g1 * g2) / (N*N) * (mu1 - mu2) * (mu1 - mu2)
    bestSeuil = 0

    # Deroulement
    for i in range(1, 255):
        mu1 = mu1 * g1 + i * histo[i]
        mu2 = mu2 * g2 - i * histo[i]

        g1 = g1 + histo[i]
        g2 = g2 - histo[i]

        mu1 = mu1 / g1
        mu2 = mu2 / g2

        if (g1 * g2) / (N*N) * (mu1 - mu2) * (mu1 - mu2) > bestVar:
            bestVar = (g1 * g2) / (N*N) * (mu1 - mu2) * (mu1 - mu2)
            bestSeuil = i

    return bestSeuil


def partageEaux(imageG, listMarq, struct):
    print ('Calcul partage des eaux')
    print ('Nombre d objets : ' + str(len(listMarq)))

    ## Phase d'initialisation
    # newImage sera notre image de sortie
    newImage = np.zeros(imageG.shape, imageG.dtype)
    # marqueurs nous permettra de ranger nos marqueurs d'entree par couleur (couleur originale) et d'assigner une couleur(couleur de partage des eaux) pour chaque marqueur
    marqueurs = [[] for i in range(0, 256)]
    # tailles stockera un nombre de pixel pour chaque nuance de gris dans le tableau marqueurs (dans le meme principe qu'un histogramme)
    tailles = np.zeros([256], np.int)

    # On parcours chaque pixel de l'image
    for i in range(0, imageG.shape[0]):
        for j in range(0, imageG.shape[1]):
            # Pour chaque marqueur en entree
            for k in range(0, len(listMarq)):
                if listMarq[k][i, j] > 0:
                    # On associe une couleur au marqueur
                    marqueurs[imageG[i, j]].append(((i, j), k + 1))
                    # On ajoute des pixels correspondant a la couleur de l'image originale dans tailles
                    tailles[imageG[i, j]] += 1
                    # On colorie les marqueurs avec la bonne couleur dans l'image de sortie
                    newImage[i, j] = k+1

                    # A ce niveau la, le tableau tailles stocke le nombre de pixel pour chaque nuance de gris present dans les marqueurs
                    print ('Initialisation terminee')

                    # Variable utilisee pour l'affichage uniquement
                    count = 0
                    # Tant qu'il reste des vale
    while np.sum(tailles) > 0:
        # Juste de l'affichage a la con
        if count % 1000 == 0:
            print('Innondation en cours. Iteration ' + str(count))
            print('numpy.sum(tailles) : ' + str(np.sum(tailles)))

        # On fait un tableau qui stocke toutes les couleurs de tailles pour lesquels on a encore des pixels (et donc nombre_de_pixel>0)
        tmp = np.where(tailles > 0)[0]
        # tmp[0] correspond a la premiere nuance de couleur pour laquelle on a encore des pixels
        # On recupere un marqueur de couleur tmp[0] (pop() supprime la derniere valeur du tableau et la retourne)
        # value correspond a la couleur de partage des eaux du marqueur
        (posx, posy), value = marqueurs[tmp[0]].pop()

        # On diminue le nombre de pixel pour la couleur tmp[0] (puisqu'on l'a retire de marqueurs)
        tailles[tmp[0]] = tailles[tmp[0]] - 1

        # !! C'est le fait d'utiliser toujours la couleur tmp[0] qui permet de faire evoluer l'innondation correctement (on commence toujours pas innonder par le bas) !!

        # On parcours notre element structurant de voisinage
        for i, j in struct:
            # On calcule la position du pixel de l'element structurant par rapport a la position du marqueur qu'on a selectionne
            x = posx + i
            y = posy + j

            # si notre pixel est dans l'image (en hauteur et en largeur)
            if x >= 0 and x < newImage.shape[0] and y >= 0 and y < newImage.shape[1] and newImage[x, y] == 0:
                # On ajoute dans les marqueurs (!! a la bonne position de nuance de gris !!) les voisins de notre pixel (ce qui permet de propager l'innondation) avec la bonne couleur de partage des eaux
                marqueurs[imageG[x, y]].append(((x, y), value))
                # On met a jour notre tableau tailles puisque marqueurs a evolue
                tailles[imageG[x, y]] += 1
                # On colorie notre image de sortie parce que sinon on a bosse pour rien
                newImage[x, y] = value
        # utilise pour l'affichage uniquement
        count += 1

    print ('Innondation terminee')

    return newImage