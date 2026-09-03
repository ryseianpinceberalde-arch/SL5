#!/usr/bin/env python
# coding: utf-8

# # 8. Make Predictions

# In[28]:


res = model.predict(X_test)


# In[29]:


actions[np.argmax(res[4])]


# In[30]:


actions[np.argmax(y_test[4])]
