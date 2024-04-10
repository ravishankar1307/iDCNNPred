%Parameters varibales which are optimized by bayesopt function using bayesian
%optimization algorithm 

 % Choose Variables to Optimize

% optimVars = [
%     optimizableVariable('filterSize',[2 10],'Type','integer')
%     optimizableVariable('filterSize2',[4 30],'Type','integer')
%     optimizableVariable('initialNumFilters',[2 16],'Type','integer')
%     optimizableVariable('InitialLearnRate',[1e-5 1e-1],'Transform','log')
%     optimizableVariable('Momentum',[0.8 0.95])
%     optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];


optimVars = [
    optimizableVariable('InitialLearnRate',[1e-5 1e-1],'Transform','log')
    optimizableVariable('Momentum',[0.8 0.95])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];

%     optimizableVariable('SectionDepth',[1 8],'Type','integer')


 %Make the objective function to minimize the validation loss by loss function

ObjFcn = makeObjFcn_Squeezenet1(imdsTrain,imdsValidation);  %working model

% Save all the results in Model_Result directory which are a current folder

current_dir=pwd;

% cd Pretrained_Model


%Perform Bayesian Optimization

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',5,...
    'MaxTime',1*60*60,...
    'IsObjectiveDeterministic',false,... 
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'UseParallel',false);


%Evaluate Final Network
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError;
cd(current_dir);

%[YPredicted,probs] = classify(savedStruct.trainedNet,XTest);
%testError = 1 - mean(YPredicted == YTest)
%Accuracy = 1-testError

% Getting  Best feasible points of optimal hyperparameter
X = bestPoint(BayesObject);