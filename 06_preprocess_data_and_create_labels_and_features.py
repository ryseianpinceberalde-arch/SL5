#!/usr/bin/env python
# coding: utf-8

# # 6. Preprocess Data and Create Labels and Features

# In[8]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[9]:


label_map = {label:num for num, label in enumerate(actions)}


# In[10]:


label_map


# In[11]:


sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


# In[12]:


np.array(sequences).shape


# In[13]:


np.array(labels).shape


# In[14]:


X = np.array(sequences)


# In[15]:


X.shape


# In[16]:


y = to_categorical(labels).astype(int)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# In[19]:


y_test.shape
