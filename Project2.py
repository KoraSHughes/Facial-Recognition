#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" Kora S. Hughes - Computer Vision: Project 2 - Facial Recognition"""
# graded on implementation, approx 20%<accuracy<100% --> PCA


# In[57]:


import numpy as np
import sys
import os
from PIL import Image
from matplotlib import pyplot as plt

np.set_printoptions(threshold=1000, edgeitems=4)

train_data = "/Face-dataset/Training/"
test_data = "/Face-dataset/Testing/"
here = os.getcwd()


# In[59]:


# training
def eigen(faces):
    """ returns an array of eigenface vectors """
    ri = np.array([face.flatten() for face in faces])  # flattened faces (1d)
    assert ri.shape[0] == len(faces)  # precaution to make sure shapes are right
    assert ri.shape[1] == faces[0].shape[0]*faces[0].shape[1]
    
    mean_f = mean_faces(ri)  # avg face
#     print("found mean faces, meu...", mean_f.shape)

    a = np.array([np.subtract(temp_face, mean_f) for temp_face in ri])  # face deviation from the mean
#     print("found a, ri vector...", a.shape)
            
#     c = np.dot(a, a.T)  # 45045 x 45045
    l = np.dot(a, a.T)  # 8x8
#     print("l shape:", l.shape)
    l_val, v = np.linalg.eig(l)  # eigenvalues and eigenvectors
#     print("eigen vec, v...:", v.shape)
    u = np.dot(a.T, v).T  # len(faces) largest eigenfaces of c
#     print("found u eigenfaces...", u.shape)  # should be 8 x 45045
    
    omega = np.array([np.dot(u, r) for r in a])  # face-space version
    return omega, u, mean_f


def mean_faces(faces):
    """ returns the mean of matricies """
    # Note: assuming all faces are the same size (a x b  or (a*b) x 1)
    mean_face = np.zeros(faces[0].shape)
    if len(faces[0].shape) == 2:  # mean of normal 2d image
        for i in range(faces[0].shape[0]):
            for j in range(faces[0].shape[1]):
                mean_face[i,j] = sum([face[i,j]/len(faces) for face in faces])
    elif len(faces[0].shape) == 1:  # mean of flattened image
        for i in range(faces[0].shape[0]):
            mean_face[i] = sum([face[i]/len(faces) for face in faces])  # np.sum(faces[:,i])/len(faces)?
    else:
        assert 0 < len(faces[0].shape) < 3
    return mean_face


def poof(flat_im, num_cols):
    """ poof up a flattened 2d image so we can see what it looks like """
    num_rows = int(len(flat_im)/num_cols)
    poof_im = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            flatify = j + i*num_cols
            poof_im[i,j] = flat_im[flatify]
    return poof_im


# In[60]:


# testing
def nn1(input_image, omega, u, meu):
    """ 1-nearest neighbor classifier 
        --> checks test image with eigenfaces and returns the closes match's index """
    I = input_image.flatten() - meu
    omega_I = np.dot(u, I)
    print("Omega_I EigenCoefficient:\n", omega_I)
    
#     Ir = np.dot(U, omega_I)
#     print("Ir shape:", Ir)
#     d0 = dist(Ir, I)
#     if d0 > t0:
#         return -1
    
    d = np.empty((len(omega)))
    for i in range(len(omega)):
        # compute euclidean distance between face and eigen-coefficients
        d[i] = dist(omega_I, omega[i])
    print("Recognition Distances:\n", d)
    return np.argmin(d)
        
    
def dist(im, coef):
    """ returns euclidean distance of im to a coefficient"""
#     print("Getting dist of", im, "&", coef)
    assert len(im) == len(coef)
    return np.sum(np.abs(im - coef))  # euclidean distance -- np.sum(np.abs(im - coef))


# In[61]:


def main():
    # Note: small amount of data
    train_files = [here+train_data+file for file in os.listdir(here+train_data)]
    test_files = [here+test_data+file for file in os.listdir(here+test_data)]
    assert len(train_files) > len(test_files) > 0
    
    # get images from directories
    train_images = []
#     print("TRAINING IMAGES")
    for file in train_files:  # run code on all files in directory
        im = Image.open(file)
        image = np.array(im)
#         plt.imshow(image, cmap='gray')  # visualize
#         plt.show()
        train_images.append(image)
    test_images = []
#     print("\n\nTEST IMAGES:")
    for file in test_files:  # run code on all files in directory
        im = Image.open(file)
        image = np.array(im)
#         plt.imshow(image, cmap='gray')  # visualize
#         plt.show()
        test_images.append(image)
    print("images imported...")
    
    # general variables about data
    m = len(train_images)
    n1 = train_images[0].shape[0]
    n2 = train_images[0].shape[1]
    print("Looking at", m, "images size", (n1,n2))
    
    # train PCA
    omega, u, meu = eigen(train_images)  # must be in order of faces (eigenface, eigencoef)
#     print("Refrence shapes:", omega.shape, u.shape, meu.shape)
    
    # show mean face:
    print("Mean Face:\n", meu)
    plt.imshow(poof(meu, n2), cmap='gray')
    plt.show()
#     add_image = Image.fromarray((poof(meu, n2)).astype(np.uint8))
#     add_image.save(here+"/mean_face.bmp")
    print('\n')
    
    # show eigenfaces
    print("EigenFaces:\n", u)
    for i in range(len(u)):
        temp_im = poof(u[i], n2)
        plt.imshow(temp_im, cmap='gray')  # visualize
        plt.show()
#         add_image = Image.fromarray((temp_im).astype(np.uint8))
#         add_image.save(here+"/eigenface"+str(i)+".bmp")
    print("EigenFace Coefficients:\n", omega)
    
    
    # test PCA
    count = 0
    for test_im in test_images:
        print("\n\n\n\n matching image:")
        plt.imshow(test_im, cmap='gray')  # visualize
        plt.show()
        print("...with...")
        ind = nn1(test_im, omega, u, meu)
        if ind == -1:
            print("Face not in training data!")
        else:
#             print("Face mapped to #"+str(ind))
            plt.imshow(train_images[ind], cmap='gray')  # visualize
            plt.show()
if __name__ == "__main__":
    print("starting...\n\n")
    main()
    print("\n\n...end")

