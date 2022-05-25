import numpy as np
import os
from numpy import loadtxt
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import glob
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

def buildData(preds, path):

    if preds.ndim == 3:
        preds = preds.reshape(preds.shape[0], preds.shape[-1])

    preds = np.argmax(preds, axis=1)

    image_class = dict()

    for num_class in range(1000):
        image_class[num_class] = []
        if num_class < 1000:
            for filename in glob.glob(path + str(num_class) + '/*.jpg'):
                image_class[num_class].append(filename)

    gts = []

    for num_class, filenames in image_class.items():
        for filename in filenames:
            gts.append(num_class)

    return preds, gts


def buildDataFromCSV():

    data = pd.read_csv('val_out_gt.csv')

    preds = data['predicted'].tolist()
    gts = data['gt'].tolist()

    return preds, gts


def computeAndSaveConfusionMatrix(gts, preds, cm_matrix_path):


    cm = confusion_matrix(gts, preds, labels=range(1000))

    np.savetxt(os.path.join(cm_matrix_path, 'cm.csv'), cm, delimiter=',')

    with open(os.path.join(cm_matrix_path, 'classification_report.txt'), 'w') as f_obj:
        f_obj.write(classification_report(gts, preds))

    return cm


def getMaxDistance(df):

    for index_row, row in df.iterrows():
        for column in df.columns.values:
            value_row = df.loc[index_row][column]
            value_column = df.loc[column][index_row]
            if value_column > value_row:
                df[index_row][column] = value_column
            else:
                df[column][index_row] = value_row

    return df



def plotDendrogram(model, categories, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=categories, distance_sort='ascending', **kwargs)


def cleanDiagonal(df):

    for num_class in range(251):
        df[num_class][num_class] = 1

    return df




def mainLoop():
    path_to_save_dendrogram = ''
    categories = [class_index for class_index in range(1000)]
    df = pd.read_csv('val_mod.csv', header=None)
    df = (df - df.min()) / (df.max() - df.min())
    df = 1 - df
    index = categories.copy()
    df.set_index = index
    df = cleanDiagonal(df)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='single')
    model = model.fit(df.values)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.figure(figsize=(50, 50))
    plotDendrogram(model, categories=categories)
    plt.savefig(path_to_save_dendrogram + 'dendrogram.pdf', dpi=1200)

if __name__ == "__main__":
    mainLoop()