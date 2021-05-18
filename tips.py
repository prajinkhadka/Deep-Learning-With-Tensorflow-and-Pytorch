# Over the Single Batch at first. 

# Dont forget to set the train or eval mode. 
# model.train() , model.eval() 

# Dont forget to .zero_grad() 

# Using softmax with cross entropy loss - Wrong 
# Crossentropy loss is , softmax and then log likelihood. 


# using bias hen using batchnorm.

# using view as permute/tranpose

import torch 
x = torch.tensor([[1,2,3], [4,5,6]])
print(x)

# Wrong 
print(x.view(3,2))

# permiute 
print(x.permute(1,0))

# Using bad data augumentation.

# Not shuffling the data. 

# Not Normalizing the data. 
# Note, ToTensor(), divides everything by 255. 


# Not slipping gradients in case of LSTM, RNN, GRU 
# after optimzier.zro_grad(), loss.backard() is
# torch.nn.utils.clip_grad_norm(model.paramters(), max_norm=value)
