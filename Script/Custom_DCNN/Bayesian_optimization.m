%Parameters varibales which are optimized by bayesopt function using bayesian optimization algorithm 

% Choose Variables to Optimize

optimVars = [
    optimizableVariable('filterSize',[2 10],'Type','integer')
    optimizableVariable('filterSize2',[4 30],'Type','integer')
    optimizableVariable('initialNumFilters',[2 16],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-5 1e-1],'Transform','log')
    optimizableVariable('Momentum',[0.8 0.95])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];


%Make the objective function to minimize the validation loss by objective loss function

ObjFcn = makeObjFcn1(imdsTrain,imdsValidation);  %working model

% Save all the results in current directory 

current_dir=pwd;


%Perform Bayesian Optimization

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',5,...
    'MaxTime',1*60*60,...
    'IsObjectiveDeterministic',false,... 
    'UseParallel',false);


%Evaluate Final Network
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError;
cd(current_dir);

% Getting  Best feasible points of optimal hyperparameter
X = bestPoint(BayesObject);