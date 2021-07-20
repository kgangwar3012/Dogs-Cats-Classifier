from imutils import paths
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import pickle



imagePaths =list(paths.list_images("C:\e"))

def image_vector(image, size=(128, 128)):
    return cv2.resize(image, size).flatten()


imagematrix = []
imagelabels = []
pixels = None
for path in imagePaths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    label = path.split(os.path.sep)[-1].split(".")[0]
    pixels = image_vector(image)

    imagematrix.append(pixels)
    if label == "dog" :
       imagelabels.append(1)
    elif label == "cat" :
       imagelabels.append(0) 

imagematrix = np.array(imagematrix)
imagelabels = np.array(imagelabels)

imagematrix = imagematrix/255.0


(X_train, X_test, Y_train, Y_test) = train_test_split(imagematrix, imagelabels, test_size=0.2, random_state=50)


pca= PCA(n_components = 800).fit(X_train)


X_train_pca =pca.transform(X_train)
X_test_pca = pca.transform(X_test)


model1 = SVC(max_iter=-1, kernel='rbf', class_weight='balanced',gamma='scale')  
model1.fit(X_train_pca, Y_train)

pickle.dump(model1, open('dogscats.pkl', 'wb'))
pickle.dump(pca, open('pca.pkl', 'wb'))

