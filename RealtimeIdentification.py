#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:30:07 2019

@author: gongchaoyun
"""

import sys
sys.path.append("../")
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from scipy import misc
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
from sklearn.svm import SVC


def face2database(picture_path,model_path,database_path,batch_size=90,image_size=160):
    #extract the features to database
    #picture_path is the path for face file
    #model_path is the path for facenet model
    #database_path is path of the face dataset
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset(picture_path)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False,image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            np.savez(database_path,emb=emb_array,lab=labels)
            print("the features dataset have been extracted ！")
            
def ClassifyTrainSVC(database_path,SVCpath):
    #database_path is the face database
    #SVCpath is the path for classifier
    Database=np.load(database_path)
    name_lables=Database['lab']
    embeddings=Database['emb']
    name_unique=np.unique(name_lables)
    labels=[]
    for i in range(len(name_lables)):
        for j in range(len(name_unique)):
            if name_lables[i]==name_unique[j]:
                labels.append(j)
    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)
    with open(SVCpath, 'wb') as outfile:
        pickle.dump((model,name_unique), outfile)
        print('Saved classifier model to file "%s"' % SVCpath)

     
def RTrecognization(facenet_model_path,SVCpath,database_path):
    #facenet_model_path is the path of facenet model
    #SVCpath is the path of SVM classifier model
    #database_path is path of the face database
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(facenet_model_path)
            with open(SVCpath, 'rb') as infile:
                    (classifymodel, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % SVCpath)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            Database=np.load(database_path)

            test_mode = "onet"
            thresh = [0.9, 0.6, 0.7]
            min_face_size = 24
            stride = 2
            slide_window = False
            shuffle = False
            detectors = [None, None, None]
            prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
            epoch = [18, 14, 16]
            model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
            PNet = FcnDetector(P_Net, model_path[0])
            detectors[0] = PNet
            RNet = Detector(R_Net, 24, 1, model_path[1])
            detectors[1] = RNet
            ONet = Detector(O_Net, 48, 1, model_path[2])
            detectors[2] = ONet
            mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
            video_capture = cv2.VideoCapture(0)
            video_capture.set(3, 800)
            video_capture.set(4, 800)
            corpbbox = None
            while True:
                 t1 = cv2.getTickCount()
                 ret, frame = video_capture.read()
                 if ret:
                    image = np.array(frame)
                    img_size=np.array(image.shape)[0:2]
                    boxes_c,landmarks = mtcnn_detector.detect(image)
                    t2 = cv2.getTickCount()
                    t = (t2 - t1) / cv2.getTickFrequency()
                    fps = 1.0 / t
                    for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]#detect face region，right-up x，left_up y，right-down x，left-down y
                        score = boxes_c[i, 4]#detect the score of face region
                        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        
                        x1=np.maximum(int(bbox[0])-16,0)
                        y1=np.maximum(int(bbox[1])-16,0)
                        x2=np.minimum( int(bbox[2])+16,img_size[1])
                        y2=np.minimum( int(bbox[3])+16,img_size[0])
                        crop_img=image[y1:y2,x1:x2]
                        scaled=misc.imresize(crop_img,(160,160),interp='bilinear')
                        img = scaled
                        img=np.reshape(img,(-1,160,160,3))
                        feed_dict = { images_placeholder:img, phase_train_placeholder:False }
                        embvecor=sess.run(embeddings, feed_dict=feed_dict)
                        embvecor=np.array(embvecor)
                        #comapre the face features with the others in database 2 by 2.
                        #tmp=np.sqrt(np.sum(np.square(embvecor-Database['emb'][0])))
                        #tmp_lable=Database['lab'][0]
                        #for j in range(len(Database['emb'])):
                            #t=np.sqrt(np.sum(np.square(embvecor-Database['emb'][j])))
                            #if t<tmp:
                                #tmp=t
                                #tmp_lable=Database['lab'][j]
                        #print(tmp)

                        # apply SVM to classify the face features
                        predictions = classifymodel.predict_proba(embvecor)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_lable=class_names[best_class_indices]
                        
                        if tmp_lable==0:
                            tmp_lable='hanwen'
                        if tmp_lable==1:
                            tmp_lable='chaoyun'
                        
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print(best_class_indices,best_class_probabilities)
                        if best_class_probabilities<0.4:
                            tmp_lable="others"
                        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                        cv2.putText(frame, '{0}'.format(tmp_lable), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
                    cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)
                    for i in range(landmarks.shape[0]):
                        for j in range(len(landmarks[i])//2):
                            cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))            
        # time end
                    cv2.imshow("", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                 else:

                    print('device not find')
                    break
            video_capture.release()
            cv2.destroyAllWindows()


            
if __name__ == "__main__":
    picture_path="/Users/gongchaoyun/Downloads/understand_facenet/data_face_160"
    model_path="/Users/gongchaoyun/Downloads/understand_facenet/20180402-114759"
    database_path="/Users/gongchaoyun/Downloads/understand_facenet/classify/Database.npz"
    SVCpath="/Users/gongchaoyun/Downloads/understand_facenet/classify/SVCmodel.pkl"
    face2database(picture_path,model_path,database_path)
    ClassifyTrainSVC(database_path,SVCpath)
    RTrecognization(model_path,SVCpath,database_path)
