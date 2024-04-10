
% Specify a large read size to minimize the cost of file I/O.

imds.ReadSize = 100;

% Set the seed of the global random number generator to aid in the reproducibility of results.

rng('default');

ImageDatasetPath = fullfile('iDCNNPred\Dataset');
imds = imageDatastore(ImageDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Use the shuffle function to shuffle the  data prior to split data.

imds = shuffle(imds);

imds.Labels;

%Split the datastore image files by function splitEachLabel 

[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds,0.8,0.1,"randomized");

% Load new imds for extrenal test and virtual screening databse as
% imdsTest_ExTest and imdsVS

ExTestDatasetPath = fullfile('E:\iDCNNPred\External_Decoy_Test_Set');
imdsExTest = imageDatastore(ExTestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


VSDatasetPath = fullfile('E:\iDCNNPred\Maybridge_VS');
imdsVS = imageDatastore(VSDatasetPath, ...
    'IncludeSubfolders',true);


% Shuffled the extrenal test set  datastore

imdsExTest = shuffle(imdsExTest);
