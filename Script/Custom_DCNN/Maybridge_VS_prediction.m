
% Predict Maybridge chemical library 

[VSPred,VSprob] = classify(trainedNet,imdsVS,'ExecutionEnvironment','gpu');

% Virtual screening hits and summary 
summary(VSPred)

Hits = [imdsVS.Files(1:end,:) VSPred(1:end,:)];

% Extract the active from the obtained hits

Ac_hits = (VSPred == 'Active');
Ac_VS = VSPred(Ac_hits,:);

% Save predictions and probabilities
T_VS = table(VSPred, VSprob, 'VariableNames', {'Prediction', 'Probabilities'});
writetable(T_VS, 'virtual_screening_predictions.csv');
