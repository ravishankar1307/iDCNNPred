% Evaluate  the training, validation, Test, ExTest accuracy and loss or error 
% Extract the trained model structure as a single varibale for further analysis 

trainedNet = savedStruct.trainedNet;

% Training Accuracy and Loss

[TrainPred,trainprobs] = classify(trainedNet,imdsTrain,"ExecutionEnvironment","gpu");
TrainingAccuracy = mean(TrainPred == imdsTrain.Labels) 
disp("training accuracy: " + TrainingAccuracy*100 + "%") 
TrainingError =  1 - mean(TrainPred == imdsTrain.Labels) 
disp("training Error: " + TrainingError + "%")

% Validation Accuracy and Loss

[ValPred,valprobs] = classify(trainedNet,imdsValidation,"ExecutionEnvironment","gpu");
ValAccuracy = mean(ValPred == imdsValidation.Labels)
disp("Validation accuracy: " + ValAccuracy*100 + "%")
ValError =  1 - mean(ValPred == imdsValidation.Labels)
disp("Validation Error: " + ValError + "%")


% Test Accuracy and Loss
[TestPred,testprobs] = classify(trainedNet,imdsTest,"ExecutionEnvironment","gpu");
TestAccuracy = mean(TestPred == imdsTest.Labels)  
disp("Test accuracy: " + TestAccuracy*100 + "%") 
TestError =  1 - mean(TestPred == imdsTest.Labels) 
disp("Test Error: " + TestError + "%")

% External Test Accuracy and Loss 

[ExTestPred,Extestprobs] = classify(trainedNet,imdsExTest,"ExecutionEnvironment","gpu");
ExTestAccuracy = mean(ExTestPred == imdsExTest.Labels)
disp("EXTest accuracy: " + ExTestAccuracy*100 + "%")
ExTestError =  1 - mean(ExTestPred == imdsExTest.Labels) 
disp("ExTest Error: " + ExTestError + "%")


% Save predictions and probabilities for training set
T_train = table(TrainPred, trainprobs, 'VariableNames', {'Prediction', 'Probabilities'});
writetable(T_train, 'training_predictions.csv');

% Save predictions and probabilities for validation set
T_validation = table(ValPred, valprobs, 'VariableNames', {'Prediction', 'Probabilities'});
writetable(T_validation, 'validation_predictions.csv');

% Save predictions and probabilities for test set
T_test = table(TestPred, testprobs, 'VariableNames', {'Prediction', 'Probabilities'});
writetable(T_test, 'test_predictions.csv');

% Save predictions and probabilities for external test set
T_exTest = table(ExTestPred, Extestprobs, 'VariableNames', {'Prediction', 'Probabilities'});
writetable(T_exTest, 'external_test_predictions.csv');






