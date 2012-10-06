--[[
Main file
By Xiang Zhang (xiang.zhang [at] nyu.edu) and Durk Kingma
(dpkingma [at] gmail.com>) @ New York University
Version 0.1, 09/22/2012

This file is implemented for the assigments of CSCI-GA.2565-001 Machine
Learning at New York University, taught by professor Yann LeCun
(yann [at] cs.nyu.edu)
]]

-- Load required libraries and files
dofile("spambase.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")

-- This is just an example
function main()
	
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(3000,1000)

   -- 2. Initialize a linear regression model with l2 regularization (lambda = 0.05)
   print("Initializing a linear regression model with l2 regularization...")
   local model = modLinReg(data_train:features(), regL2(0.05))
   local model_percep = modPercep(data_train:features(), regL2(0.05))
   local model_log_reg = modLogReg(data_train:features(), regL2(0.05))
   
   -- 3. Initialize a batch trainer with constant step size = 0.05
   print("Initializing a batch trainer with constant step size 0.05...")
   local trainer = trainerBatch(model, stepCons(0.05))
   local trainer_percep_stoch = trainerSGD(model_percep, stepCons(0.05))
   local trainer_log_reg_stoch = trainerSGD(model_log_reg, stepCons(0.05))
   
   -- 4. Perform batch training for 100 steps
   print("Training for 100 batch steps...")
   local loss_train, error_train = trainer:train(data_train, 100)
   local loss_train_percep, error_train_percep = trainer_percep_stoch:train(data_train, 100)
   local loss_train_log_reg, error_train_log_reg = trainer_log_reg_stoch:train(data_train, 100)

   -- 5. Perform test using the model
   print("Testing...")
   local loss_test, error_test = trainer:test(data_test)
   local loss_test_percep, error_test_percep = trainer_percep_stoch:test(data_test)
   local loss_test_log_reg, error_test_log_reg = trainer_log_reg_stoch:test(data_test)

   -- 6. Print the result
   print("Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)
   print("\n")
   print("For PERCEPTRON: Training loss = "..loss_train_percep..", error = "..error_train_percep.."; Testing loss = "..loss_test_percep..", error = "..error_test_percep)
   print("\n")
   -- print("For DIRECT SOLUTION: Training loss = Training error = "..model:train(data_train))
   print("\n")
   print("FOR LOGISTIC REGRESSION: Training loss = "..loss_train_log_reg..", error = "..error_train_log_reg.."; Testing loss = "..loss_test_log_reg..", error = "..error_test_log_reg)
end

main()
