
train_labels = double(nominal(imdsTrain.Labels));
val_labels = double(nominal(imdsValidation.Labels));
test_labels = double(nominal(imdsTest.Labels));
Extest_labels = double(nominal(imdsExTest.Labels));
% Extest_labels = double(nominal(imdsExtest.Labels));


[train_fp_rate, train_tp_rate, ~, train_AUC] = perfcurve(train_labels, trainprobs(:,1), 1);
[val_fp_rate, val_tp_rate, ~, val_AUC] = perfcurve(val_labels, valprobs(:,1), 1);
[test_fp_rate, test_tp_rate, ~, test_AUC] = perfcurve(test_labels, testprobs(:,1), 1);
[extest_fp_rate, extest_tp_rate, ~, extest_AUC] = perfcurve(Extest_labels, Extestprobs(:,1), 1);

[train_precision, train_recall, ~, train_AUC_PRC] = perfcurve(train_labels, trainprobs(:,1), 1, 'xCrit', 'reca', 'yCrit', 'prec');
[val_precision, val_recall, ~, val_AUC_PRC] = perfcurve(val_labels, valprobs(:,1), 1, 'xCrit', 'reca', 'yCrit', 'prec');
[test_precision, test_recall, ~, test_AUC_PRC] = perfcurve(test_labels, testprobs(:,1), 1, 'xCrit', 'reca', 'yCrit', 'prec');
[extest_precision, extest_recall, ~, extest_AUC_PRC] = perfcurve(Extest_labels, Extestprobs(:,1), 1, 'xCrit', 'reca', 'yCrit', 'prec');


% Create a table with the AUC values
datasets = {'train', 'val', 'test', 'Extest'};
AUC_values = [train_AUC, train_AUC_PRC; val_AUC, val_AUC_PRC; test_AUC, test_AUC_PRC; extest_AUC, extest_AUC_PRC];
AUC_table = array2table(AUC_values, 'VariableNames', {'ROC_AUC', 'PRC_AUC'}, 'RowNames', datasets);

% Write the table to a CSV file
writetable(AUC_table, 'AUC_metrics.csv', 'WriteRowNames', true);