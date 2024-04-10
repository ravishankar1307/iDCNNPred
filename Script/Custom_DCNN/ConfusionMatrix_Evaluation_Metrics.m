% Plot the confusion matrix. 

%Training confusion matrix
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(imdsTrain.Labels,TrainPred);
cm.Title = 'Confusion Matrix for 2D-Image CNN Training Set';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';


%Validaion confusion matrix 

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(imdsValidation.Labels,ValPred);
cm.Title = 'Confusion Matrix for 2D-Image CNN Validation Set';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%Test confusion matrix 
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(imdsTest.Labels,TestPred);
cm.Title = 'Confusion Matrix for 2D-Image CNN test Set';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%External Test confusion matrix
figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(imdsExTest.Labels,ExTestPred);
cm.Title = 'Confusion Matrix for 2D-Image CNN External test Set';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%Evalaution metrics for training set
CM.train = confusionmat(imdsTrain.Labels,TrainPred);
% Here first find the all true positive value using diag function 
 CM.tp_m = diag(CM.train);
% each class find the TP, TN, FP, FN using the following code

                    TP=CM.train(1,1);
                    FN=CM.train(1,2);
                    FP=CM.train(2,1);
                    TN=CM.train(2,2);

                  
               
    CM.Accuracy = (TP+TN)./(TP+FP+TN+FN)

    TPR = TP./(TP + FN);%tp/actual positive  RECALL SENSITIVITY
    if isnan(TPR)
        TPR = 0;
    end
    PPV = TP./ (TP + FP); % tp / predicted positive PRECISION
    if isnan(PPV)
        PPV = 0;
    end
    TNR = TN./ (TN+FP); %tn/ actual negative  SPECIFICITY
    if isnan(TNR)
        TNR = 0;
    end
    FPR = FP./ (TN+FP);
    if isnan(FPR)
        FPR = 0;
    end
    FScore = (2*(PPV * TPR)) / (PPV+TPR);

    if isnan(FScore)
        FScore = 0;
    end

CM.TPR = TPR;
CM.PPV = PPV;
CM.TNR =TNR;
CM.FPR = FPR;
CM.FScore =FScore;

CM.TP = TP;
CM.FN = FN;
CM.FP = FP;
CM.TN = TN;

CM.BAcc = (TPR + TNR) ./2;

% Matthew correlation coefficient(MCC) value calculate 

CM.MCC = (TP .* TN - FP .* FN) ./ ...
    sqrt( (TP + FP) .* (TP + FN) .* (TN + FP) .* (TN + FN) );

CM.gmean = sqrt(TPR*TNR);
% CM.kappa = cohensKappa(imdsTrain.Labels, TrainPred);
DPX = TPR ./(1-TPR);
DPY = TNR ./(1-TNR);
CM.DP = sqrt(3)./pi*log(DPX)+log(DPY);


% Means of metric ; TPR = True Positive Rate, TNR = True Negative Rate,
% PPV = Positive Predictive Value, NPV = Negative Predictive Value.

% precision or PPV  = tp / (tp + fp);
% recall or sensitivity or TPR  = TP / (TP + FN);
% specificity or TNR = TN / (TN+FP);
% FPR = FP / (TN+FP);
% FScore = (2*(PPV * TPR)) / (PPV+TPR);


% #####################################################################################################################################

%Evalaution metrics for training set
CM.train = confusionmat(imdsTrain.Labels,TrainPred);
% Here first find the all true positive value using diag function 
 CM.tp_m = diag(CM.train);
% each class find the TP, TN, FP, FN using the following code

                    TP=CM.train(1,1);
                    FN=CM.train(1,2);
                    FP=CM.train(2,1);
                    TN=CM.train(2,2);

                  
                    
    CM.Accuracy = (TP+TN)./(TP+FP+TN+FN)

    TPR = TP./(TP + FN);%tp/actual positive  RECALL SENSITIVITY
    if isnan(TPR)
        TPR = 0;
    end
    PPV = TP./ (TP + FP); % tp / predicted positive PRECISION
    if isnan(PPV)
        PPV = 0;
    end
    TNR = TN./ (TN+FP); %tn/ actual negative  SPECIFICITY
    if isnan(TNR)
        TNR = 0;
    end
    FPR = FP./ (TN+FP);
    if isnan(FPR)
        FPR = 0;
    end
    FScore = (2*(PPV * TPR)) / (PPV+TPR);

    if isnan(FScore)
        FScore = 0;
    end

CM.TPR = TPR;
CM.PPV = PPV;
CM.TNR =TNR;
CM.FPR = FPR;
CM.FScore =FScore;

CM.TP = TP;
CM.FN = FN;
CM.FP = FP;
CM.TN = TN;

CM.BAcc = (TPR + TNR) ./2;

% Matthew correlation coefficient(MCC) value calculate 

CM.MCC = (TP .* TN - FP .* FN) ./ ...
    sqrt( (TP + FP) .* (TP + FN) .* (TN + FP) .* (TN + FN) );

CM.gmean = sqrt(TPR*TNR);
DPX = TPR ./(1-TPR);
DPY = TNR ./(1-TNR);
CM.DP = sqrt(3)./pi*log(DPX)+log(DPY);
CM.kappa = cohensKappa(imdsTrain.Labels, TrainPred);


% Save training and validation metrics to CSV
T_train = struct2table(CM, 'AsArray', true);
writetable(T_train, 'train_metrics.csv');


% Evaluation metrics for validation set
CM.validation = confusionmat(imdsValidation.Labels, ValPred);
CM.tp_m_validation = diag(CM.validation);
TP_validation = CM.validation(1,1);
FN_validation = CM.validation(1,2);
FP_validation = CM.validation(2,1);
TN_validation = CM.validation(2,2);
CM.Accuracy_validation = (TP_validation + TN_validation) / (TP_validation + FP_validation + TN_validation + FN_validation);
TPR_validation = TP_validation / (TP_validation + FN_validation);
if isnan(TPR_validation)
    TPR_validation = 0;
end
PPV_validation = TP_validation / (TP_validation + FP_validation);
if isnan(PPV_validation)
    PPV_validation = 0;
end
TNR_validation = TN_validation / (TN_validation + FP_validation);
if isnan(TNR_validation)
    TNR_validation = 0;
end
FPR_validation = FP_validation / (TN_validation + FP_validation);
if isnan(FPR_validation)
    FPR_validation = 0;
end
FScore_validation = (2 * (PPV_validation * TPR_validation)) / (PPV_validation + TPR_validation);
if isnan(FScore_validation)
    FScore_validation = 0;
end
CM.TPR_validation = TPR_validation;
CM.PPV_validation = PPV_validation;
CM.TNR_validation = TNR_validation;
CM.FPR_validation = FPR_validation;
CM.FScore_validation = FScore_validation;
CM.TP_validation = TP_validation;
CM.FN_validation = FN_validation;
CM.FP_validation = FP_validation;
CM.TN_validation = TN_validation;
CM.BAcc_validation = (TPR_validation + TNR_validation) / 2;
CM.MCC_validation = (TP_validation * TN_validation - FP_validation * FN_validation) / sqrt((TP_validation + FP_validation) * (TP_validation + FN_validation) * (TN_validation + FP_validation) * (TN_validation + FN_validation));
CM.gmean_validation = sqrt(TPR_validation * TNR_validation);
DPX_validation = TPR_validation / (1 - TPR_validation);
DPY_validation = TNR_validation / (1 - TNR_validation);
CM.DP_validation = sqrt(3) / pi * log(DPX_validation) + log(DPY_validation);
CM.kappa_validation = cohensKappa(imdsValidation.Labels, ValPred);


% Save validation metrics to CSV
T_validation = struct2table(CM, 'AsArray', true);
writetable(T_validation, 'validation_metrics.csv');


% Evaluation metrics for test set
CM.test = confusionmat(imdsTest.Labels, TestPred);
CM.tp_m_test = diag(CM.test);
TP_test = CM.test(1,1);
FN_test = CM.test(1,2);
FP_test = CM.test(2,1);
TN_test = CM.test(2,2);
CM.Accuracy_test = (TP_test + TN_test) / (TP_test + FP_test + TN_test + FN_test);
TPR_test = TP_test / (TP_test + FN_test);
if isnan(TPR_test)
    TPR_test = 0;
end
PPV_test = TP_test / (TP_test + FP_test);
if isnan(PPV_test)
    PPV_test = 0;
end
TNR_test = TN_test / (TN_test + FP_test);
if isnan(TNR_test)
    TNR_test = 0;
end
FPR_test = FP_test / (TN_test + FP_test);
if isnan(FPR_test)
    FPR_test = 0;
end
FScore_test = (2 * (PPV_test * TPR_test)) / (PPV_test + TPR_test);
if isnan(FScore_test)
    FScore_test = 0;
end
CM.TPR_test = TPR_test;
CM.PPV_test = PPV_test;
CM.TNR_test = TNR_test;
CM.FPR_test = FPR_test;
CM.FScore_test = FScore_test;
CM.TP_test = TP_test;
CM.FN_test = FN_test;
CM.FP_test = FP_test;
CM.TN_test = TN_test;
CM.BAcc_test = (TPR_test + TNR_test) / 2;
CM.MCC_test = (TP_test * TN_test - FP_test * FN_test) / sqrt((TP_test + FP_test) * (TP_test + FN_test) * (TN_test + FP_test) * (TN_test + FN_test));
CM.gmean_test = sqrt(TPR_test * TNR_test);
DPX_test = TPR_test / (1 - TPR_test);
DPY_test = TNR_test / (1 - TNR_test);
CM.DP_test = sqrt(3) / pi * log(DPX_test) + log(DPY_test);
CM.kappa_test = cohensKappa(imdsTest.Labels, TestPred);


% Save test metrics to CSV
T_test = struct2table(CM, 'AsArray', true);
writetable(T_test, 'test_metrics.csv');


% Evaluation metrics for ExTest set
CM.exTest = confusionmat(imdsExTest.Labels, ExTestPred);
CM.tp_m_exTest = diag(CM.exTest);
TP_exTest = CM.exTest(1,1);
FN_exTest = CM.exTest(1,2);
FP_exTest = CM.exTest(2,1);
TN_exTest = CM.exTest(2,2);
CM.Accuracy_exTest = (TP_exTest + TN_exTest) / (TP_exTest + FP_exTest + TN_exTest + FN_exTest);
TPR_exTest = TP_exTest / (TP_exTest + FN_exTest);
if isnan(TPR_exTest)
    TPR_exTest = 0;
end
PPV_exTest = TP_exTest / (TP_exTest + FP_exTest);
if isnan(PPV_exTest)
    PPV_exTest = 0;
end
TNR_exTest = TN_exTest / (TN_exTest + FP_exTest);
if isnan(TNR_exTest)
    TNR_exTest = 0;
end
FPR_exTest = FP_exTest / (TN_exTest + FP_exTest);
if isnan(FPR_exTest)
    FPR_exTest = 0;
end
FScore_exTest = (2 * (PPV_exTest * TPR_exTest)) / (PPV_exTest + TPR_exTest);
if isnan(FScore_exTest)
    FScore_exTest = 0;
end
CM.TPR_exTest = TPR_exTest;
CM.PPV_exTest = PPV_exTest;
CM.TNR_exTest = TNR_exTest;
CM.FPR_exTest = FPR_exTest;
CM.FScore_exTest = FScore_exTest;
CM.TP_exTest = TP_exTest;
CM.FN_exTest = FN_exTest;
CM.FP_exTest = FP_exTest;
CM.TN_exTest = TN_exTest;
CM.BAcc_exTest = (TPR_exTest + TNR_exTest) / 2;
CM.MCC_exTest = (TP_exTest * TN_exTest - FP_exTest * FN_exTest) / sqrt((TP_exTest + FP_exTest) * (TP_exTest + FN_exTest) * (TN_exTest + FP_exTest) * (TN_exTest + FN_exTest));
CM.gmean_exTest = sqrt(TPR_exTest * TNR_exTest);
DPX_exTest = TPR_exTest / (1 - TPR_exTest);
DPY_exTest = TNR_exTest / (1 - TNR_exTest);
CM.DP_exTest = sqrt(3) / pi * log(DPX_exTest) + log(DPY_exTest);
CM.kappa_exTest = cohensKappa(imdsExTest.Labels, ExTestPred);

% Save ExTest metrics to CSV
T_exTest = struct2table(CM, 'AsArray', true);
writetable(T_exTest, 'exTest_metrics.csv');

