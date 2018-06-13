import os
import random

import cv2
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.externals import joblib

def get_files(data_path):
    print('#### Scanning dataset')
    samples = pd.DataFrame(columns=['label', 'path'])

    for folder in os.listdir(data_path):
        print('\tReading image category: {}'.format(folder))
        files = []

        for img in os.listdir(os.path.join(data_path, folder)):
            files.append(os.path.join(data_path, folder, img))

        files = pd.DataFrame(data=files, columns=['path'])
        files.insert(0, 'label', folder)
        samples = samples.append(files, ignore_index=True)        
    
    return(samples)


def extract_descriptors(samples):
    print('#### Extracting SURF descriptors')
    surf = cv2.xfeatures2d_SURF.create(4000, 5, 5, False, False)
    descriptors = pd.DataFrame()

    for idx, row in samples.iterrows():
        print('\tProcessing {} / {}'.format(idx+1, len(samples)))
        img = cv2.imread(row['path'])
        _, descriptor = surf.detectAndCompute(img, None)
        descriptor = pd.DataFrame(descriptor)
        descriptor.insert(0, 'source', idx)
        descriptors = descriptors.append(descriptor, ignore_index=True)
    
    print('#### Consolidating dataset')
    descriptors = pd.merge(samples, descriptors, how='right', left_index=True, right_on='source')
    return(descriptors)


def create_classifier(samples, labels):
    classifier = SVC()
    classifier.fit(samples, labels)
    return(classifier)


if __name__ == "__main__":

    new_dataset = False
    new_training_data = False
    new_codebook = True
    new_classifier = True

    data = None

    if new_dataset:
        data = get_files('.\\images')
        data = extract_descriptors(data)
        print('#### Saving dataset')
        data.to_csv('./dataset.csv')

    if new_training_data:
        if data is None:
            print('#### Loading dataset')
            data = pd.read_csv('./dataset.csv')

        print('#### Creating training sets')
        data['label'].to_csv('data_labels.csv', index=False, header=False, sep=';')
        data[data.columns[-64:]].to_csv('data_features.csv', index=False, header=False, sep=';')
    
    if new_codebook:
        print('#### Creating codebook and vector representations')
        os.system('java -jar openXBOW.jar -i data_features.csv -o data_bow.csv -size 200 -B web_app/model/model_codebook')

    if new_classifier:
        samples = pd.read_csv('./data_bow.csv', header=None, sep=';')
        print(samples.head())
        labels = pd.read_csv('./data_labels.csv', header=None, sep=';')
        print(labels.head())
        classifier = create_classifier(samples, labels)
        print('#### Saving classifier')
        joblib.dump(classifier, './web_app/model/model_classifier.svc')
        