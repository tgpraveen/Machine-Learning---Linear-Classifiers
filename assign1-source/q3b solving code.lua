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
function tempmain()
   local lambda = {0.03,0.05,0.07,0.09}
   local train_size = {10,30,100}
   for j = 1, 3 do
print("\nFor training set size "..train_size[j]..":")
for i = 1,4 do
   -- 1. Load spambase dataset
   
   local data_train, data_test = spambase:getDatasets(train_size[j],1000)

   -- 2. Initialize a linear regression model with l2 regularization (lambda = 0.05)
  
   local model_log_reg = modLogReg(data_train:features(), regL2(lambda[i]))
   local model_log_reg_l1 = modLogReg(data_train:features(), regL1(lambda[i]))
   
   -- 3. Initialize a batch trainer with constant step size = 0.05
 
   local trainer_log_reg_stoch = trainerSGD(model_log_reg, stepCons(0.05))
   local trainer_log_reg_stoch_l1 = trainerSGD(model_log_reg_l1, stepCons(0.05))
   
   -- 4. Perform batch training for 100 steps
  
   local loss_train_log_reg, error_train_log_reg = trainer_log_reg_stoch:train(data_train, 100)
   local loss_train_log_reg_l1, error_train_log_reg_l1 = trainer_log_reg_stoch_l1:train(data_train, 100)

   -- 5. Perform test using the model
 
   local loss_test_log_reg, error_test_log_reg = trainer_log_reg_stoch:test(data_test)
   local loss_test_log_reg_l1, error_test_log_reg_l1 = trainer_log_reg_stoch_l1:test(data_test)

   -- 6. Print the result
   
   print("Lambda = "..lambda[i].."\n\tl2 reg: Training loss = "..loss_train_log_reg..", error = "..error_train_log_reg.."; Testing loss = "..loss_test_log_reg..", error = "..error_test_log_reg)
   -- print("\n")
   print("\tl1 reg: Training loss = "..loss_train_log_reg_l1..", error = "..error_train_log_reg_l1.."; Testing loss = "..loss_test_log_reg_l1..", error = "..error_test_log_reg_l1)
  end  
  end
end

tempmain()
