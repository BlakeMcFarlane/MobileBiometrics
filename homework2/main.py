

'''
    <START SET UP>
    Suppress warnings and import necessary libraries.
    Import code for loading data and extracting features.
'''

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import matplotlib.pyplot as plt
import numpy as np
import math 

# k-Nearest Neighbors - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
'''
    <END SET UP>
'''

'''
    Load facial landmarks (5 or 68)
'''
X = np.load("X-5-SoF.npy")
y = np.load("y-5-SoF.npy")
num_identities = y.shape[0]


'''
    Transform landmarks into features
'''
def triangle_area(p1, p2, p3):
    return 0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))

features = []
for k in range(num_identities):
    person_k = X[k]
    # Coordinate points for each landmark
    out_left_eye = person_k[0]
    in_left_eye = person_k[1]
    out_right_eye = person_k[2]
    in_right_eye = person_k[3]
    mouth = person_k[4]

    features_k = []

    # Nose Area
    nose_triangle_area = triangle_area(in_left_eye, in_right_eye, mouth)
    features_k.append(nose_triangle_area)

    # Left Eye and Mouth Area
    right_face_area = triangle_area(out_right_eye, in_right_eye, mouth)
    features_k.append(right_face_area)

    # Right Eye and Mouth Area
    left_face_area = triangle_area(in_left_eye, out_left_eye, mouth)
    features_k.append(left_face_area)

    # Other Significant facial landmarks
    centroid_x = (out_left_eye[0] + out_right_eye[0] + mouth[0]) / 3
    centroid_y = (out_left_eye[1] + out_right_eye[1] + mouth[1]) / 3
    features_k.append(centroid_x)  
    features_k.append(centroid_y) 
    features_k.append(mouth[0]) 
    features_k.append(mouth[1]) 
    features_k.append(out_right_eye[0]) 
    features_k.append(out_right_eye[1]) 
    features_k.append(out_left_eye[0]) 
    features_k.append(out_left_eye[1]) 

    # Ensure float division
    eyes_width_ratio = (abs(float(out_right_eye[0]) - float(in_right_eye[0])) + abs(float(out_left_eye[0]) - float(in_left_eye[0]))) / float(abs(in_left_eye[0] - in_right_eye[0]))
    features_k.append(eyes_width_ratio - 1)  

    features.append(features_k)

features = np.array(features)


''' 
    Create an instance of the classifier
'''

clf = RandomForestClassifier(class_weight='balanced')

num_correct = 0
num_incorrect = 0

# Create empty lists to store results for plotting
pred_labels = []  # Predictions from classifier
actual_labels = []  # Actual labels


for i in range(0, len(y)):
    query_X = features[i, :]
    query_y = y[i]
    
    template_X = np.delete(features, i, 0)
    template_y = np.delete(y, i)
        
    # Set the appropriate labels
    # 1 is genuine, 0 is impostor
    y_hat = np.zeros(len(template_y))
    y_hat[template_y == query_y] = 1 
    y_hat[template_y != query_y] = 0
    
    # Train the classifier
    clf.fit(template_X, y_hat) 
    
    # Predict the label of the query
    y_pred = clf.predict(query_X.reshape(1,-1)) 
    
    # Store the prediction and actual label for visualization
    pred_labels.append(y_pred[0])
    actual_labels.append(1 if query_y in template_y else 0)
    
    # Get results
    if y_pred == 1:
        num_correct += 1
    else:
        num_incorrect += 1

# Print results
print()
print("Num correct = %d, Num incorrect = %d, Accuracy = %0.2f" 
      % (num_correct, num_incorrect, num_correct/(num_correct+num_incorrect))) 

plt.grid(True)
plt.show()  