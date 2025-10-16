#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NAME : JANARTHANAN K
# Reg.No: 212223040072


# In[2]:


# expt-6-edge detection-sobel,laplacian,canny

import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Load the image
image = cv2.imread('lion.jpg')  # Replace with your image path


# In[4]:


# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[5]:


# Original Image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')


# In[6]:


# ------------------ Sobel Edge Detection ------------------
# Detect edges in X and Y directions

sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Sobel in x direction
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Sobel in y direction
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Combine both directions
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')


# In[7]:


# ------------------ Laplacian Edge Detection ------------------
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.axis('off')


# In[9]:


# ------------------ Canny Edge Detection ------------------
canny_edges = cv2.Canny(gray_image, 50, 150)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')  


# In[ ]:




