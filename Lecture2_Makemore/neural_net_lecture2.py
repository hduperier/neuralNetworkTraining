#!/usr/bin/env python
# coding: utf-8

# In[31]:


# Makemore (takes input, makes more of it)
# Character level leanguage modeling
import torch
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


words = open('names.txt', 'r').read().splitlines()


# In[33]:


words[:10]


# In[34]:


len(words)


# In[35]:


min(len(w) for w in words)


# In[36]:


max(len(w) for w in words)


# In[37]:


for w in words[:1]:
    for ch1, ch2 in zip(w, w[1:]):
        print(ch1, ch2)


# In[38]:


# Starting with Bigram Language Model
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1


# In[39]:


sorted(b.items(), key = lambda kv: -kv[1])


# In[40]:


N = torch.zeros((27, 27), dtype=torch.int32)
N


# In[41]:


# makes a massive string of the names and makes and list and sorts
chars = sorted(list(set(''.join(words))))
# lookup table
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


# In[42]:


for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


# In[43]:


plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');


# In[45]:


# First row, all columns
N[0,:]


# In[46]:


# convert to probabilities
p = N[0].float()

p = p/p.sum()
p


# In[69]:


g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
itos[ix]


# In[70]:


g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
p


# In[71]:


torch.multinomial(p, num_samples=100, replacement=True, generator=g)


# In[76]:


g = torch.Generator().manual_seed(2147483647)

for i in range(20):
    
    out = []
    ix = 0
    while True:
        p = N[ix].float()
        p = p/p.sum()
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out))


# In[ ]:




