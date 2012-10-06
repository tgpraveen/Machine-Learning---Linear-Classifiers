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
		  local W_star = {}
		  local X = torch.zeros(dataset:size(),dataset:features())
		  local Y = torch.zeros(dataset:size())
		  -- local test = {1}
		  for i = 1,dataset:size() do
			for j = 1,dataset:features() do				
	--			print(dataset[1][1][1])
--				print("i "..i.." j "..j)
		  		X[i][j] = dataset[i][1][j]
		  		Y[i] = dataset[i][2][1]
			end 	
		  end
		  print("Hi\n")
		  print(X[1])
		  X_T = X:transpose(1,2)
		  local Z = torch.mm(X_T,X)
		  -- local Z = torch.mm(X,X:t())	
		  W_star = torch.mm(torch.mm(torch.inverse(Z),X:transpose(1,2)),Y)
		  -- local error_train = 
	return 	

   end
   -- Return this model
   return model
end

-- Perceotron module: f(x) = w^T x
-- inputs: dimension of inputs; r: a regularizer
function modPercep(inputs, r)
   local model = {}
   -- Generate weight vector initialized randomly
   model.w = torch.rand(inputs)
   -- Define the loss function. Output is areal number (not 1-dim tensor!)
   -- Assuming y is a 1-dim tensor. Taking regularizer into consideration
   function model:l(x,y)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  -- return (torch.dot(model.w,x) - y[1])^2/2 + r:l(model.w)
	  return ((model:g(x)[1]-y[1])*torch.dot(model.w,x)) + r:l(model.w)
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
		return 	x*(-y[1]+model:g(x)[1]) + r:dw(model.w)
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
		return (2*torch.log(1+torch.exp(-y[1]*(torch.dot(model.w:transpose(1,2),x)))))
   end
   -- Define the gradient function. Taking regularizer into consideration.
   function model:dw(x,y)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  -- return 	x*(-y[1]+model:g(x)[1]) + r:dw(model.w)
	     return (torch.dot(-(y[1]-g(torch.dot(model.w:transpose(1,2),x))),x))
   end
   -- Define the output function. Output is a 1-dim tensor.
   function model:f(x)
      -- Remove the following line and add your stuff
      -- print("You have to define this function by yourself!");
	  -- return torch.ones(1)*torch.dot(model.w,x)
		 return ((torch.ones(1)*torch.exp(torch.dot(model.w:transpose(1,2),x))-1)/(torch.exp(torch.dot(model.w:transpose(1,2),x))+1))
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
   print("You have to define this function by yourself!");
end
