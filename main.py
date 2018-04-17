import cv2, kMean, utils, strel

img = cv2.imread('Images/caillou.png')
utils.displayImage(img, 'Image initiale')

# On applique kmean
imgKMean = kMean.reducColor(img, 6)
utils.displayImage(imgKMean, 'Image kmean')

# On nettoie le resultat obtenu (on enleve les zones trop petites)
imgray = cv2.cvtColor(imgKMean, cv2.COLOR_BGR2GRAY)

# imgray = cv2.medianBlur(imgray,5)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)

# Get the results
# The first cell is the number of labels
num_labels = output[0]
print num_labels
# The second cell is the label matrix
labels = output[1]
# The third cell is the stat matrix
stats = output[2]
# The fourth cell is the centroid matrix
centroids = output[3]

imgNettoyee = imgKMean

elementOuverture = strel.build('carre', 2, None)
elementReconstruction = strel.build('carre', 2, None)
#
# imgNettoyee = utils.fermetureReconstruction(imgNettoyee, elementOuverture, elementReconstruction)
# imgNettoyee = utils.ouvertureReconstruction(imgNettoyee, elementOuverture, elementReconstruction)

taille = imgKMean.shape
for x in range(0, taille[0]):
    for y in range(0, taille[1]):
        if stats[labels[x][y], cv2.CC_STAT_AREA] < 800 or stats[labels[x][y], cv2.cv2.CC_STAT_WIDTH] < 30:
            imgNettoyee[x, y] = imgNettoyee[x-1, y]


# imgNettoyee = kMean.nettoyageImage(img, pixelMax, largeurMax)
utils.displayImage(imgNettoyee, 'Image nettoyee')

# On extrait les contours de l'image
imgContour = kMean.recupContour(img, imgNettoyee)

# imgray = cv2.cvtColor(imgKMean, cv2.COLOR_BGR2GRAY)
# th3 = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# im2, contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# imgContour = img
# imgContour[:] = (255, 255, 255)
#
# cv2.drawContours(imgContour, contours, -1, (0, 0, 0), 2)

utils.displayImage(imgContour, 'Image contour')

# On pose les indications dans les zones
imgray = cv2.cvtColor(imgContour, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)

output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
centroids = output[3]

structElement = strel.build_as_list('disque', 2, None)

for centroid in centroids[0:output[0]]:
    for (i, j) in structElement:
        if int(centroid[1]) < (taille[1] - 2) and int(centroid[0]) < (taille[0] - 2):
            imgContour[i + int(centroid[1]), j + int(centroid[0])] = imgNettoyee[int(centroid[1]), int(centroid[0])]

# Affichage du resultat final
utils.displayImage(imgContour, 'fin')