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
function mainq2d()
	
   -- 1. Load spambase dataset
   print("Initializing datasets...")
    local data_train, data_test = spambase:getDatasets(10,1000)
    local data_train2, data_test_unused1 = spambase:getDatasets(30,1000)
	local data_train3, data_test_unused2 = spambase:getDatasets(100,1000)
	local data_train4, data_test_unused3 = spambase:getDatasets(500,1000)
	local data_train5, data_test_unused4 = spambase:getDatasets(1000,1000)
	local data_train6, data_test_unused5 = spambase:getDatasets(3000,1000)

   -- 2. Initialize a linear regression model with l2 regularization (lambda = 0.05)
   -- print("Initializing a linear regression model with l2 regularization...")
   
	local model_log_reg = modLogReg(data_train:features(), regL2(0.05)) --l2 reg
	local model_log_reg2 = modLogReg(data_train2:features(), regL2(0.05)) --l2 reg
	local model_log_reg3 = modLogReg(data_train3:features(), regL2(0.05)) --l2 reg
	local model_log_reg4 = modLogReg(data_train4:features(), regL2(0.05)) --l2 reg
	local model_log_reg5 = modLogReg(data_train5:features(), regL2(0.05)) --l2 reg
	local model_log_reg6 = modLogReg(data_train6:features(), regL2(0.05)) --l2 reg
   
   
   -- 3. Initialize a batch trainer with constant step size = 0.05
   -- print("Initializing a batch trainer with constant step size 0.05...")
   
	local trainer_log_reg_stoch = trainerSGD(model_log_reg, stepCons(0.1)) --l2 reg
	local trainer_log_reg_stoch2 = trainerSGD(model_log_reg2, stepCons(0.1)) --l2 reg
	local trainer_log_reg_stoch3 = trainerSGD(model_log_reg3, stepCons(0.1)) --l2 reg
	local trainer_log_reg_stoch4 = trainerSGD(model_log_reg4, stepCons(0.05)) --l2 reg
	local trainer_log_reg_stoch5 = trainerSGD(model_log_reg5, stepCons(0.03)) --l2 reg
	local trainer_log_reg_stoch6 = trainerSGD(model_log_reg6, stepCons(0.01)) --l2 reg


      
   -- 4. Perform batch training for x steps
   
   print("\n10 steps Logistic regression convergence testing:")
   local loss_train_log_reg, error_train_log_reg = trainer_log_reg_stoch:train(data_train, 1000)
   print ("------------------------------\n\n")
   print("\n30 steps Logistic regression convergence testing:")
   local loss_train_log_reg2, error_train_log_reg2 = trainer_log_reg_stoch2:train(data_train2, 1000)
   print ("------------------------------\n\n")
   print("\n100 steps Logistic regression convergence testing:")
   local loss_train_log_reg3, error_train_log_reg3 = trainer_log_reg_stoch3:train(data_train3, 1000)
   print ("------------------------------\n\n")
   print("\n500 steps Logistic regression convergence testing:")
   local loss_train_log_reg4, error_train_log_reg4 = trainer_log_reg_stoch4:train(data_train4, 1000)
   print ("------------------------------\n\n")
   print("\n1000 steps Logistic regression convergence testing:")
   local loss_train_log_reg5, error_train_log_reg5 = trainer_log_reg_stoch5:train(data_train5, 1000)
   print ("------------------------------\n\n")
   print("\n3000 steps Logistic regression convergence testing:")
   local loss_train_log_reg6, error_train_log_reg6 = trainer_log_reg_stoch6:train(data_train6, 1000)
   print ("------------------------------\n\n")
  
   
   -- 5. Perform test using the model
   print("Testing all against SAME Test Set......")
   
	local loss_test_log_reg, error_test_log_reg = trainer_log_reg_stoch:test(data_test)
	local loss_test_log_reg2, error_test_log_reg2 = trainer_log_reg_stoch2:test(data_test)
	local loss_test_log_reg3, error_test_log_reg3 = trainer_log_reg_stoch3:test(data_test)
	local loss_test_log_reg4, error_test_log_reg4 = trainer_log_reg_stoch4:test(data_test)
	local loss_test_log_reg5, error_test_log_reg5 = trainer_log_reg_stoch5:test(data_test)
	local loss_test_log_reg6, error_test_log_reg6 = trainer_log_reg_stoch6:test(data_test)

   -- 6. Print the result
   
   print("10 FOR LOGISTIC REGRESSION with l2 reg: Training loss = "..loss_train_log_reg..", error = "..error_train_log_reg.."; Testing loss = "..loss_test_log_reg..", error = "..error_test_log_reg)
print("30 FOR LOGISTIC REGRESSION with l2 reg: Training loss = "..loss_train_log_reg2..", error = "..error_train_log_reg2.."; Testing loss = "..loss_test_log_reg2..", error = "..error_test_log_reg2)
print("100 FOR LOGISTIC REGRESSION with l2 reg: Training loss = "..loss_train_log_reg3..", error = "..error_train_log_reg3.."; Testing loss = "..loss_test_log_reg3..", error = "..error_test_log_reg3)
print("500 FOR LOGISTIC REGRESSION with l2 reg: Training loss = "..loss_train_log_reg4..", error = "..error_train_log_reg4.."; Testing loss = "..loss_test_log_reg4..", error = "..error_test_log_reg4)
print("1000 FOR LOGISTIC REGRESSION with l2 reg: Training loss = "..loss_train_log_reg5..", error = "..error_train_log_reg5.."; Testing loss = "..loss_test_log_reg5..", error = "..error_test_log_reg5)
print("3000 FOR LOGISTIC REGRESSION with l2 reg: Training loss = "..loss_train_log_reg6..", error = "..error_train_log_reg6.."; Testing loss = "..loss_test_log_reg6..", error = "..error_test_log_reg6)
   end

mainq2d()
