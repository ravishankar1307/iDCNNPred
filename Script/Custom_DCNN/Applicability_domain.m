
% Feature extraction from trained model train, validation, test, External test, and FDA data 

layer = 'conv_5';
featuresTrain = activations(trainedNet1,imdsTrain,layer);
featuresVal = activations(trainedNet1,imdsValidation,layer);
featuresTest = activations(trainedNet1,imdsTest,layer);
featuresFDA = activations(trainedNet1,imdsFDA,layer);

whos featuresTrain

% labels of train, validation and test set
yTrain= imdsTrain.Labels;
yVal=imdsValidation.Labels;
yTest=imdsTest.Labels;

% transpose the 4D to 2D 
featuresTrain = squeeze(mean(featuresTrain,[1 2]))';
featuresVal = squeeze(mean(featuresVal,[1 2]))';
featuresTest = squeeze(mean(featuresTest,[1 2]))';
featuresFDA = squeeze(mean(featuresFDA,[1 2]))';


Mdl_knn = fitcknn(featuresTrain, yTrain, 'NumNeighbors', 5, 'Standardize', 1, 'Distance', 'euclidean');

[~, dist_val] = knnsearch(featuresTrain, featuresVal, 'K', 5, 'Distance', 'euclidean');

[~, dist_test] = knnsearch(featuresTrain, featuresTest, 'K', 5, 'Distance', 'euclidean');

[~, dist_train] = knnsearch(featuresTrain, featuresTrain, 'K', 5, 'Distance', 'euclidean');

[~, dist_fda] = knnsearch(featuresTrain, featuresFDA, 'K', 5, 'Distance', 'euclidean');



% Define a threshold for the applicability domain
threshold = 0.5; % replace with your threshold

% Determine whether each validation set data point is within the applicability domain
isWithinAD_val = all(dist_val <= threshold, 2);

% Determine whether each test set data point is within the applicability domain
isWithinAD_test = all(dist_test <= threshold, 2);

% Determine whether each train set data point is within the applicability domain
isWithinAD_train = all(dist_train <= threshold, 2);


% Determine whether each FDA set data point is within the applicability domain
isWithinAD_FDA = all(dist_fda <= threshold, 2);

% Perform PCA on the validation set features
[coeff_val, score_val, ~] = pca(featuresVal);

% Perform PCA on the test set features
[coeff_test, score_test, ~] = pca(featuresTest);

% Perform PCA on the train set features
[coeff_train, score_train, ~] = pca(featuresTrain);


% Perform PCA on the FDA set features
[coeff_fda, score_fda, ~] = pca(featuresFDA);


% Take the first two principal components for the validation set
reducedFeatures_val = score_val(:, 1:2);

% Take the first two principal components for the test set
reducedFeatures_test = score_test(:, 1:2);

% Take the first two principal components for the test set
reducedFeatures_train = score_train(:, 1:2);


% Take the first two principal components for the test set
reducedFeatures_fda = score_fda(:, 1:2);


% Create a combined scatter plot for the training, validation, test, and FDA sets
figure;
hold on;
h_train = gscatter(reducedFeatures_train(:, 1), reducedFeatures_train(:, 2), isWithinAD_train, 'yk', '.', 10, 'off');
h_val = gscatter(reducedFeatures_val(:, 1), reducedFeatures_val(:, 2), isWithinAD_val, 'br', '.', 10, 'off');
h_test = gscatter(reducedFeatures_test(:, 1), reducedFeatures_test(:, 2), isWithinAD_test, 'mg', '.', 10, 'off');
h_fda = gscatter(reducedFeatures_fda(:, 1), reducedFeatures_fda(:, 2), isWithinAD_FDA, [0.5 0 0.5; 1 0.5 0], '.', 10, 'off');
hold off;
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Combined Applicability Domain Visualization');
legend([h_train(1), h_val(1), h_test(1), h_fda(1)], 'Inside AD (Training)', 'Inside AD (Validation)', 'Inside AD (Test)', 'Outside AD (FDA)');
legend('Location', 'best'); % Adjust the location of the legend
legend('show'); % Show the legend

% Count the number of data points inside and outside the AD for Train set
num_inside_AD_train = sum(isWithinAD_train);
num_outside_AD_train = sum(~isWithinAD_train);

% Print the counts for Train set
fprintf('Number of Train data points inside the applicability domain: %d\n', num_inside_AD_train);
fprintf('Number of Train data points outside the applicability domain: %d\n', num_outside_AD_train);

% Count the number of data points inside and outside the AD for Validation set
num_inside_AD_val = sum(isWithinAD_val);
num_outside_AD_val = sum(~isWithinAD_val);

% Print the counts for Validation set
fprintf('Number of Validation data points inside the applicability domain: %d\n', num_inside_AD_val);
fprintf('Number of Validation data points outside the applicability domain: %d\n', num_outside_AD_val);

% Count the number of data points inside and outside the AD for Test set
num_inside_AD_test = sum(isWithinAD_test);
num_outside_AD_test = sum(~isWithinAD_test);

% Print the counts for Test set
fprintf('Number of Test data points inside the applicability domain: %d\n', num_inside_AD_test);
fprintf('Number of Test data points outside the applicability domain: %d\n', num_outside_AD_test);

% Count the number of data points inside and outside the AD for FDA set
num_inside_AD_FDA = sum(isWithinAD_FDA);
num_outside_AD_FDA = sum(~isWithinAD_FDA);

% Print the counts for FDA set
fprintf('Number of FDA data points inside the applicability domain: %d\n', num_inside_AD_FDA);
fprintf('Number of FDA data points outside the applicability domain: %d\n', num_outside_AD_FDA);






