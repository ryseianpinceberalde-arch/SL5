#!/usr/bin/env python
# coding: utf-8

# # 10. Evaluation using Confusion Matrix and Accuracy

# In[33]:


from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# In[41]:


yhat = model.predict(X_test)


# In[42]:


ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


# In[43]:


multilabel_confusion_matrix(ytrue, yhat)


# In[44]:


accuracy_score(ytrue, yhat)
