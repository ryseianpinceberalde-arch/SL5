#!/usr/bin/env python
# coding: utf-8

# # 9. Save Weights

# In[31]:


model.save('action.h5')


# In[217]:


del model


# In[14]:


model.load_weights('action.h5')
