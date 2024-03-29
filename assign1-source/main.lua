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
dofile("mnist.lua")
dofile("model.lua")
dofile("regularizer.lua")
dofile("trainer.lua")

-- This is just an example
function main()
	
   -- 1. Load spambase dataset
   print("Initializing datasets...")
   local data_train, data_test = spambase:getDatasets(3000,1000)
   local data_train_multinom, data_test_multinom = mnist:getDatasets(6000,1000)

   -- 2. Initialize a linear regression model with l2 regularization (lambda = 0.05)
   -- print("Initializing a linear regression model with l2 regularization...")
   local model = modLinReg(data_train:features(), regL2(0.05))
   local model_direct_l2 = modLinReg(data_train:features(), regL2(0.05))
   model_direct_l2:train(data_train)
   local model_direct_l1 = modLinReg(data_train:features(), regL1(0.05))
   model_direct_l1:train(data_train)
   local model_lin_reg_stoch = modLinReg(data_train:features(), regL2(0.05))
   local model_percep = modPercep(data_train:features(), regL2(0.05))
   local model_log_reg = modLogReg(data_train:features(), regL2(0.05)) --l2 reg
   local model_log_reg_l1 = modLogReg(data_train:features(), regL1(0.05))
   local multinom_log_reg = modMulLogReg(data_train_multinom:features(),10,regL2(0.05))
   
   -- 3. Initialize a batch trainer with constant step size = 0.05
   -- print("Initializing a batch trainer with constant step size 0.05...")
   local trainer = trainerBatch(model, stepCons(0.05))
   local trainer_direct_l2 = trainerBatch(model_direct_l2, stepCons(0.05))
   local trainer_direct_l1 = trainerBatch(model_direct_l1, stepCons(0.05))

   local trainer_lin_stoch = trainerSGD(model_lin_reg_stoch, stepCons(0.05))

   local trainer_percep_stoch = trainerBatch(model_percep, stepCons(0.05))
   local trainer_log_reg_stoch = trainerSGD(model_log_reg, stepCons(0.05)) --l2 reg
   local trainer_log_reg_stoch_l1 = trainerSGD(model_log_reg_l1, stepCons(0.05))	
   local trainer_multinom_log_reg = trainerSGD(multinom_log_reg, stepCons(0.05))

      
   -- 4. Perform batch training for 100 steps
  -- print("Training for programmed batch steps...")
   local loss_train, error_train = trainer:train(data_train, 100)
   local loss_train_direct_l2, error_train_direct_l2 = trainer_direct_l2:test(data_train, 100)
   local loss_train_direct_l1, error_train_direct_l1 = trainer_direct_l1:test(data_train, 100)
   -- print ("\n\nThe loss for linear regression is:")
   local loss_train_lin_stoch, error_train_lin_stoch = trainer_lin_stoch:train(data_train, 100)
   -- print ("------------------------------\n\n")
   local loss_train_percep, error_train_percep = trainer_percep_stoch:train(data_train, 200)
   --print("\nLogistic regression convergence testing:")
   local loss_train_log_reg, error_train_log_reg = trainer_log_reg_stoch:train(data_train, 3000)
   --print ("------------------------------\n\n")
   local loss_train_log_reg_l1, error_train_log_reg_l1 = trainer_log_reg_stoch_l1:train(data_train, 100)
   local loss_train_multinom_log_reg, error_train_multinom_log_reg = trainer_multinom_log_reg:train(data_train_multinom, 100)
   
   -- 5. Perform test using the model
   print("Testing...")
   local loss_test, error_test = trainer:test(data_test)
   local loss_test_lin_stoch, error_test_lin_stoch = trainer_lin_stoch:test(data_test)
   local loss_test_percep, error_test_percep = trainer_percep_stoch:test(data_test)
   local loss_test_log_reg, error_test_log_reg = trainer_log_reg_stoch:test(data_test)
   local loss_test_log_reg_l1, error_test_log_reg_l1 = trainer_log_reg_stoch_l1:test(data_test)
   local loss_test_multinom_log_reg, error_test_log_multinom_reg = trainer_multinom_log_reg:test(data_test_multinom)

   -- 6. Print the result
   print("For Linear regression with Batch trainer Training loss = "..loss_train..", error = "..error_train.."; Testing loss = "..loss_test..", error = "..error_test)
   print("\n")
   print("For Linear regression with Direct solution with l2 loss = "..loss_train_direct_l2..", error = "..error_train_direct_l2..";")
   print("\n")
   print("For Linear regression with Direct solution with l1 loss = "..loss_train_direct_l1..", error = "..error_train_direct_l1..";")
   print("\n")
   print("For Linear regression with SGD Training loss = "..loss_train_lin_stoch..", error = "..error_train_lin_stoch.."; Testing loss = "..loss_test_lin_stoch..", error = "..error_test_lin_stoch)
   print("\n")
   print("For PERCEPTRON: Training loss = "..loss_train_percep..", error = "..error_train_percep.."; Testing loss = "..loss_test_percep..", error = "..error_test_percep)
   print("\n")
   print("FOR LOGISTIC REGRESSION with l2 reg: Training loss = "..loss_train_log_reg..", error = "..error_train_log_reg.."; Testing loss = "..loss_test_log_reg..", error = "..error_test_log_reg)
   -- print("\n")
   print("FOR LOGISTIC REGRESSION with l1 reg: Training loss = "..loss_train_log_reg_l1..", error = "..error_train_log_reg_l1.."; Testing loss = "..loss_test_log_reg_l1..", error = "..error_test_log_reg_l1)
   print("FOR MULTINOMIAL REGRESSION with l2 reg: Training loss = "..loss_train_multinom_log_reg..", error = "..error_train_multinom_log_reg.."; Testing loss = "..loss_test_multinom_log_reg..", error = "..error_test_log_multinom_reg)
end

main()
