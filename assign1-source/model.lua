--[[
Models implementation
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com) @ New York University
Version 0.1, 09/24/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)

In this file you should implement models satisfying the following convention,
so that they can be used by trainers you will implement in trainer.lua.

A model object consists of the following fields:

model.w: the parameter tensor. Will be updated by a trainer.

model:l(x,y): the loss function. Should take regularization into consideration.
Should assume that x and y are both tensors. The return value must be a
scalar number (not 1-dim tensor!)

model:dw(x,y): the gradient function. Should take regularization into
consideration. Should assume that x and y are both tensors. The return value
is a tensor of the same dimension as model.w

model:f(x): the output function. Depending on the model, the output function
is the output of a model prior to passing it into a decision function. For
example, in linear model f(x) = w^T x, or in logistic regression model
f(x) = (exp(w^T x) - 1)/(exp(w^T x) + 1). The output should be a tensor.

model:g(x): the decision function. This will produce a vector that will match
the labels. For example, in binary classification it should return either [1]
or [-1] (usually a thresholding by f(x)). The output should be a tensor. This
output will be used in a trainer to test the error rate of a model.

model:train(datasets, ...): (optional) direct training. If a model can be
directly trained using a closed-form formula, it can be implemented here
so that we do not need any trainer for it. Additional parameter is at your
choice (e.g., regularization).

The way I would recommend you to program the model above is to write a func-
tion which returns a table containing the fields above. As an example, a
linear regression model (modLinReg) is provided.

For additional information regarding regularizer, please refer to
regularizer.lua.

For additional information regarding the trainer, please refer to trainer.lua

]]

-- Linear regression module: f(x) = w^T x
-- inputs: dimension of inputs; r: a regularizer
function modLinReg(inputs, r)
   local model = {}
   -- Generate a weight vector initialized randomly
   model.w = torch.rand(inputs)
   -- Define the loss function. Output is a real number (not 1-dim tensor!).
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      return (torch.dot(model.w,x) - y[1])^2/2 + r:l(model.w)
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      return x*(torch.dot(model.w,x) - y[1]) + r:dw(model.w)
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      return torch.ones(1)*torch.dot(model.w,x)
   end
   -- Define the indicator function, who gives a binary classification
   function model:g(x)
      if model:f(x)[1] >= 0 then return torch.ones(1) end
      return -torch.ones(1)
   end
   -- Train directly without a trainer. Should return average loss and
   -- error on the training data
   function model:train(dataset)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  --  Andrew Ng's video lecture: Normal Equation.
	  --  local W_star = {}
	  --  local X = torch.zeros(dataset:size(),dataset:features())
	  --  local Y = torch.zeros(dataset:size(),1)
		  -- local test = {1}
		--  for i = 1,dataset:size() do
	  -- for j = 1,dataset:features() do				
	--			print(dataset[1][1][1])
--				print("i "..i.." j "..j)
		 -- 		X[i][j] = dataset[i][1][j]
		  	--	Y[i] = dataset[i][2][1]
		--	end 	
		 -- end
		--  print("Hi\n")
		  -- print(X[1])

		  -- X_T = X:transpose(1,2)
		  -- local A_side = torch.mm(X_T,X)
		  -- local Z = torch.mm(X,X:t())
		  -- Inverse method:
		     -- local A_side_inverse = torch.inverse(A_side)
		     -- W_star = torch.mm(torch.mm(A_side_inverse,X:transpose(1,2)),Y)
		  -- Try to solve using AX=B torch.gesv()
		  -- local B_side = torch.mm(X_T,Y)
		  -- W_star = torch.gesv(A_side, B_side)
		  -- local error_train = 
	--return 	

   --end
   -- Return this model
   --return model
	  local s = dataset:size()
      local feature_matrix = torch.Tensor(s,dataset:features())
      local target_vector = torch.Tensor(s,1)
      for i = 1,s do
        feature_matrix[i] = dataset[i][1]
        target_vector[i] = dataset[i][2]
      end
      local trans = feature_matrix:t()
      local xtx = torch.mm(trans,feature_matrix)
      local inv = torch.inverse(xtx)
      model.w = torch.mm(torch.mm(inv,trans),target_vector)
   end
   -- Return this model
   return model

end

-- Perceotron module: f(x) = w^T x
-- inputs: dimension of inputs; r: a regularizer
function modPercep(inputs, r)
   local model = {}
   -- Generate weight vector initialized randomly
   model.w = torch.zeros(inputs)
   -- Define the loss function. Output is areal number (not 1-dim tensor!)
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  -- return (torch.dot(model.w,x) - y[1])^2/2 + r:l(model.w)
	  return ((model:g(x)[1]-y[1])*torch.dot(model.w,x)) --+ r:l(model.w)
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
		return 	x*(-y[1]+ model:g(x)[1]) --+ r:dw(model.w)
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
		return torch.ones(1)*torch.dot(model.w,x)
   end
   -- Define the indicator function, who gives a binary classification
   function model:g(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
		if model:f(x)[1] >= 0 then return torch.ones(1) end
      		return -torch.ones(1)
   		end
   -- Return this model
   return model
end

-- Logistic regression module: f(x) = (exp(w^T x) - 1)/(exp(w^T x) + 1)
-- inputs: dimension of inputs; r: a regularizer
function modLogReg(inputs, r)
   -- Remove the following line and add your stuff
   -- print("You have to define this function by yourself!");
	local model = {}
   -- Generate weight vector initialized randomly
   model.w = torch.rand(inputs)
   -- Define the loss function. Output is areal number (not 1-dim tensor!)
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  -- return (torch.dot(model.w,x) - y[1])^2/2 + r:l(model.w)
	  -- return ((model:g(x)[1]-y[1])*torch.dot(model.w,x)) + r:l(model.w)
		return 2*torch.log(1+torch.exp(-y[1]*(torch.dot(model.w,x)))) + r:l(model.w)
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  -- return 	x*(-y[1]+model:g(x)[1]) + r:dw(model.w)
	  -- print("dw of x: ")
	  -- print(((model:g(x))))
	     return (x*((model:g(x)[1]-y[1])) + r:dw(model.w))
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  -- return torch.ones(1)*torch.dot(model.w,x)
	  -- local dot_prod = torch.dot(model.w,x)
	  -- print("hi")
	  -- print(x)
		 return (torch.ones(1)*(((torch.exp(torch.dot(model.w,x)))-1)/(torch.exp(torch.dot(model.w,x))+1)))
   end
   -- Define the indicator function, who gives a binary classification
   function model:g(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
		if model:f(x)[1] >= 0 then return torch.ones(1) end
      		return -torch.ones(1)
   		end
   -- Return this model
   return model
end

-- Multinomial logistic regression module f(x)_k = exp(w_k^T x) / (\sum_j exp(w_j^Tx))
-- inputs: dimension of inputs; classes: number of classes; r: a regularizer
function modMulLogReg(inputs, classes, r)
   -- Remove the following line and add your stuff
   -- print("You have to define this function by yourself!");
	 local model = {}
   -- Generate a weight vector initialized randomly
   model.w = torch.rand(classes, inputs) 
   -- Define the loss function. Output is a real number (not     1-dim tensor!).
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      a = 0
      b = y[1]
      for i = 1, classes do
      a = a + torch.exp(torch.dot(model.w[i],x)) 
      end   
      return (torch.log(a/(torch.exp(torch.dot(model.w[b],x))))) + r:l(model.w)
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      a = 0
      b = y[1]
      for i = 1, classes do
      a = a + torch.exp(torch.dot(model.w[i],x)) 
      end 
      m = torch.exp(torch.dot(model.w[b],x))
      lw = torch.zeros(model.w:size())
      for i = 1, classes do
        if i == b then lw[i] = x*(m/a-1) else lw[i] = x*(m/a) end
      end
      return (lw + r:dw(model.w))
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      v = torch.ones(classes)  
      a = 0     
      for i = 1, classes do
         a = a + torch.exp(torch.dot(model.w[i],x))
      end
      for i = 1, classes do
           v[i] = (torch.exp(torch.dot(model.w[i],x)))/a
      end
      return v

   end
   -- Define the indicator function, who gives a binary classification
   function model:g(x)
      max = 0 
      index = 1 
      v = model:f(x)
      for i = 1, classes do
          if v[i] >= max then 
          max = v[i]
          index =i 
          end    
      end
      return torch.ones(1)*index
   end
   -- Return this model
   return model
end
