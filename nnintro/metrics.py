import numpy as np
from math import sqrt
import pandas as pd
import json


def removeTrain(predDR, predHM):
    df = pd.read_csv("/mnt/nfs-data/public/xkristof/labels_val.csv")
    keypoints = np.load("/mnt/nfs-data/public/xkristof/keypoints.npy")
    predDRList = list()
    predHMList = list()
    keypointsList = list()
    for i in range(len(df)):
        inde = df.iloc[i]["Unnamed: 0"]
        predDRList.append(predDR[inde])
        predHMList.append(predHM[inde])
        keypointsList.append(keypoints[inde])
    predDRKeypoints = np.array(predDRList)
    predHMKeypoints = np.array(predHMList)
    keypoints = np.array(keypointsList)
    predDRKeypoints = predDRKeypoints.astype("float64")
    predHMKeypoints = predHMKeypoints.astype("float64")

    return predDRKeypoints, predHMKeypoints, keypoints

# Compute Euclidean distance
def get_distance(a, b):
    xa = a[0]
    ya = a[1]
    xb = b[0]
    yb = b[1]
    return sqrt((abs(xa - xb))**2+(abs(ya-yb))**2)

def is_correct(pred, groundtruth, threshold):
    if groundtruth[2] == 0:
        if pred[2] == 0:
            return True
        else:
            return False
    if get_distance(pred, groundtruth) < threshold and pred[2] == groundtruth[2]:
        return True
    return False

# Gets PCK for one pose
def PCKPose(predPose, groundtruthPose, threshold):
    nCorrect = 0
    for i in range(14):
        if is_correct(predPose[i], groundtruthPose[i], threshold):
            nCorrect += 1
    return nCorrect / 14

# Gets PCK for multiple poses
def PCK(predKeypoints, keypoints, threshold):
    pckList = []
    for poseI in range(predKeypoints.shape[0]):
        pckMeasure = PCKPose(predKeypoints[poseI], keypoints[poseI], threshold)
        pckList.append(pckMeasure)
    return sum(pckList) / len(pckList)

def isLimbCorrect(predPose, groundtruthPose, connection):

    # each limb is "made up" of three parts
    # eg. arm is made up of shoulder, elbow and hand
    predLimb = (predPose[connection[0]], predPose[connection[1]], predPose[connection[2]])
    groundtruthLimb = (groundtruthPose[connection[0]], groundtruthPose[1], groundtruthPose[connection[2]])

    half_the_limb = 0.5 * (get_distance(groundtruthLimb[0], groundtruthLimb[1]) + get_distance(groundtruthLimb[1], groundtruthLimb[2]))# polovicna vzdialenost medzi dvomi groundtruth jointmi

    is_first_correct = is_correct(predLimb[0], groundtruthLimb[0], half_the_limb)
    is_second_correct = is_correct(predLimb[2], groundtruthLimb[2], half_the_limb)
    return is_first_correct and is_second_correct

# Gets PCP for one pose
def PCPPose(predPose, groundtruthPose):
    # indices of limb keypoints
    connections = [(0, 1, 2), (5, 4, 3), (6, 7, 8), (11, 10, 9)]
    nCorrect = 0

    for connection in connections:
        if isLimbCorrect(predPose, groundtruthPose, connection):
            nCorrect += 1
    return nCorrect / len(connections)

# Gets PCP for multiple poses
def PCP(predKeypoints, keypoints):
    pcpList = []
    for poseI in range(predKeypoints.shape[0]):
        pcpMeasure = PCPPose(predKeypoints[poseI], keypoints[poseI])
        pcpList.append(pcpMeasure)
    return sum(pcpList) / len(pcpList)

def PDJPose(predPose, groundtruthPose, fraction, boundingBox):
    nCorrect = 0 
    topleft = (boundingBox["x1"] / 128, boundingBox["y1"] / 128)
    bottomright = (boundingBox["x2"] / 128, boundingBox["y2"] / 128)
    
    scale = get_distance(topleft, bottomright)

    for i in range(14):
        if is_correct(predPose[i], groundtruthPose[i], fraction * scale):
            nCorrect += 1
    return nCorrect / 14

def PDJ(predKeypoints, keypoints, fraction):
    with open("/mnt/nfs-data/public/xkristof/yolo.json", 'r') as file:
        boundingBoxes = json.load(file)
    file.close()

    df = pd.read_csv("/mnt/nfs-data/public/xkristof/labels_val.csv")
    # actual indices of poses in .npy files
    valIndices = list(df.get("Unnamed: 0"))

    newBoundingBoxes = list()
    for key in boundingBoxes:
        
        # Get an int out of a leading zero format
        k = key[-10:-4].lstrip("0")
        try:
            k = int(k)
        except ValueError: # zero found
            k = 0
        if k in valIndices:
            newBoundingBoxes.append(boundingBoxes[key])

    boundingBoxes = newBoundingBoxes

    pdjList = []
    for poseI in range(predKeypoints.shape[0]):
        pdjMeasure = PDJPose(predKeypoints[poseI], keypoints[poseI], fraction, boundingBoxes[poseI])
        pdjList.append(pdjMeasure)
    return sum(pdjList) / len(pdjList)

# Gets confusion matrix values for one keypoint
def get_confusion_matrix_values_keypoints(predKeypoint, groundtruthKeypoint, threshold):
    distance = get_distance(predKeypoint, groundtruthKeypoint)
    if groundtruthKeypoint[2] == 0 and predKeypoint[2] == 1: # not visible
        return np.array([0, 0, 1, 0]) # false negative
    if groundtruthKeypoint[2] == 0 and predKeypoint[2] == 0:
        return np.array([0, 0, 0, 0]) # not adding anything
    if distance <= threshold:
        return np.array([1, 0, 0, 0]) # true positive
    if distance > threshold:
        return np.array([0, 1, 0, 0]) # false positive
    
# Gets confusion matrix values for one pose
def get_confusion_matrix_values_keypoints_poses(predPose, groundtruthPose, threshold):
    result = np.zeros(4)
    for i in range(14): # iterate through keypoints
        result += get_confusion_matrix_values_keypoints(predPose[i], groundtruthPose[i], threshold)
    return result

# Gets confusion matrix values for multiple poses
def get_confusion_matrix_values(predKeypoints, keypoints, threshold):
    result = np.zeros(4)
    for poseI in range(predKeypoints.shape[0]):
        result += get_confusion_matrix_values_keypoints_poses(predKeypoints[poseI], keypoints[poseI], threshold)
    return result

# Computes precision
def get_precision(predKeypoints, keypoints, threshold):
    confusion_matrix = get_confusion_matrix_values(predKeypoints, keypoints, threshold)
    tp = confusion_matrix[0]
    fp = confusion_matrix[1]
    if (tp+fp) == 0:
        return 0
    return tp / (tp + fp)

# Computes recall
def get_recall(predKeypoints, keypoints, threshold):
    confusion_matrix = get_confusion_matrix_values(predKeypoints, keypoints, threshold)
    tp = confusion_matrix[0]
    fn = confusion_matrix[2]
    if (tp+fn) == 0:
        return 0
    return tp / (tp + fn)