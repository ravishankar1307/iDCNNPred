% Retrain new model with optimal hyperparameter sets

        imageSize =  [200,200,3];
        Labels = categories(imdsTrain.Labels);
        numClasses = numel(categories(imdsTrain.Labels));

  % Specify the class weights in the same order as the classes appear in categories(imdsTrain). 
  % To give each class equal total weight in the loss, use class weights that are inversely proportional to the number of training examples in each class.      
        
        classes = ["Active" "Inactive"];
        classWeights = 1./countcats(imdsTrain.Labels);
        classWeights = classWeights'/mean(classWeights);

        %initialNumFilters = round((max(imageSize)/2)/sqrt(optVars.NetworkDepth));
        numMaxPools=3;
        PoolSizeAvg = floor(max(imageSize)/(2^(numMaxPools)));
        %filterSize = 5;

% Define Neural Network Architecture layers

layers1 = [
                imageInputLayer(imageSize,'Name','input')
    
                convolution2dLayer(X.filterSize,X.filterSize2,'Padding','same','Name','conv_1') %3,8
                batchNormalizationLayer('Name','BN_1');
                reluLayer('Name','relu_1');   
    
                maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_1')
    
                convolution2dLayer(X.filterSize,2*X.filterSize2,'Padding','same','Name','conv_2') %3,16
                batchNormalizationLayer('Name','BN_2');
                reluLayer('Name','relu_2');   
    
                maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_2')
                
                convolution2dLayer(X.filterSize,4*X.filterSize2,'Padding','same','Name','conv_3') %3,32
                batchNormalizationLayer('Name','BN_3')
                reluLayer('Name','relu_3')   
    
                maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_3')
%                 dropoutLayer(0.50,'Name','Drop_1')
                convolution2dLayer(X.filterSize,8*X.filterSize2,'Padding','same','Name','conv_4') %3,32
                batchNormalizationLayer('Name','BN_4')
                reluLayer('Name','relu_4') 
    
                maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_4') % uncomment conv5
     
                convolution2dLayer(X.filterSize,16*X.filterSize2,'Padding','same','Name','conv_5')%3,32
                batchNormalizationLayer('Name','BN_5')
                reluLayer('Name','relu_5') 
        

                additionLayer(2,'Name','add')
                averagePooling2dLayer(2,'Stride',2,'Name','avpoo1')
              fullyConnectedLayer(4096,'Name','FC_1')
                reluLayer('Name','relu_6')
                dropoutLayer(0.50,'Name','drop_3')
              fullyConnectedLayer(4096,'Name','FC_2')
                reluLayer('Name','relu_7')
                dropoutLayer(0.50,'Name','drop_4')
                %  fullyConnectedLayer(4096,'Name','FC_3')
                % reluLayer('Name','relu_8')
                % dropoutLayer(0.50,'Name','drop_5')
                fullyConnectedLayer(numClasses,'Name','FC')
                softmaxLayer('Name','softmax')
               classificationLayer('Name','ClassOut','Classes',Labels,'ClassWeights',classWeights)];  


layers2 = [
                convolution2dLayer(X.filterSize,X.filterSize2,'Padding','same','Name','l2_conv_1') %3,8
                batchNormalizationLayer('Name','l2_BN_1');
                reluLayer('Name','l2_relu_1');   
    
                 maxPooling2dLayer(2,'Stride',2,'Name','l2_MaxPool_1')
    
                convolution2dLayer(X.filterSize,2*X.filterSize2,'Padding','same','Name','l2_conv_2') %3,16
                batchNormalizationLayer('Name','l2_BN_2');
                reluLayer('Name','l2_relu_2');   
    
                 maxPooling2dLayer(2,'Stride',2,'Name','l2_MaxPool_2')
    
                convolution2dLayer(X.filterSize,4*X.filterSize2,'Padding','same','Name','l2_conv_3') %3,32
                batchNormalizationLayer('Name','l2_BN_3')
                reluLayer('Name','l2_relu_3')   
    
                maxPooling2dLayer(2,'Stride',2,'Name','l2_MaxPool_3')
%                 dropoutLayer(0.5,'Name','Drop_2')
                convolution2dLayer(X.filterSize,8*X.filterSize2,'Padding','same','Name','l2_conv_4') %3,32
                 batchNormalizationLayer('Name','l2_BN_4')
                reluLayer('Name','l2_relu_4') 
    
                maxPooling2dLayer(2,'Stride',2,'Name','l2_MaxPool_4') % change uncommnet conv5
     
                convolution2dLayer(X.filterSize,16*X.filterSize2,'Padding','same','Name','l2_conv_5') %3,32
                batchNormalizationLayer('Name','l2_BN_5')
                reluLayer('Name','l2_relu_5')  

];

lgraph_new = layerGraph(layers1);

lgraph_new = addLayers(lgraph_new,layers2);
lgraph_new = connectLayers(lgraph_new,'input','l2_conv_1');
lgraph_new = connectLayers(lgraph_new,'l2_relu_5','add/in2');

%figure; plot(lgraph)


        
        
%         layers = [
%             imageInputLayer(imageSize)
%             
%             % The spatial input and output sizes of these convolutional
%             % layers are 32-by-32, and the following max pooling layer
%             % reduces this to 16-by-16.
%             convBlock(optVars.filterSize,initialNumFilters,optVars.NetworkDepth)
%             maxPooling2dLayer(2,'Stride',2) 
%             % 1. maxPool
%             
%             % The spatial input and output sizes of these convolutional
%             % layers are 16-by-16, and the following max pooling layer
%             % reduces this to 8-by-8.
%             convBlock(optVars.filterSize,2*initialNumFilters,optVars.NetworkDepth)
%             maxPooling2dLayer(2,'Stride',2) 
%             % 2. maxPool
%             
%             % The spatial input and output sizes of these convolutional
%             % layers are 8-by-8. The global average pooling layer averages
%             % over the 8-by-8 inputs, giving an output of size
%             % 1-by-1-by-4*initialNumFilters. With a global average
%             % pooling layer, the final classification output is only
%             % sensitive to the total amount of each feature present in the
%             % input image, but insensitive to the spatial positions of the
%             % features.
%             convBlock(optVars.filterSize,4*initialNumFilters,optVars.NetworkDepth)
%             maxPooling2dLayer(2,'Stride',2)   
%             % 3. maxPool
%             
%             convBlock(optVars.filterSize,8*initialNumFilters,optVars.NetworkDepth)
%             %averagePooling2dLayer(PoolSizeAvg)
%             
%             % Add the fully connected layer and the final softmax and
%             % classification layers.
%             fullyConnectedLayer(numClasses)
%             softmaxLayer
%             classificationLayer];



%Train Network 
% Specify the training options. Use the sgdm optimizer with a mini-batch size of 128. Train for 30 epochs and reduce the learning rate by a factor of 10 after 5 epochs.
            miniBatchSize = 128;
%             validationFrequency =floor(numel(imdsTrain.Labels)/miniBatchSize);
            validationFrequency = floor(numel(categories(imdsTrain.Labels))/miniBatchSize);
            if validationFrequency<1
                validationFrequency=1;
            end
          
      gpuDevice(1);
     
        options1 = trainingOptions('sgdm',...
            'Momentum',X.Momentum, ... 
            'InitialLearnRate',X.InitialLearnRate,...
            'LearnRateSchedule','piecewise',...
            'LearnRateDropFactor',0.2,...
            'LearnRateDropPeriod',10,...
            'L2Regularization',X.L2Regularization, ...
            'GradientThresholdMethod','l2norm', ...
            'GradientThreshold',4, ...
            'MaxEpochs',20, ...
            'MiniBatchSize',miniBatchSize,...
            'Verbose',false, ...
            'VerboseFrequency',1, ...
            'ValidationData',imdsValidation,...
            'ValidationFrequency',validationFrequency, ...
            'ValidationPatience',Inf, ...
            'Shuffle','every-epoch',...
            'ExecutionEnvironment','gpu',...
            'Plots','training-progress',...
            'BatchNormalizationStatistics','population', ...
            'OutputNetwork','best-validation-loss');
           
                
           
         
%             'CheckpointPath','F:\Predictive_DL_model\Checkpoints');

%          'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3), ...

        
%         imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-5,5], ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3]);
%  
%         datasource = augmentedImageDatastore(imageSize,imdsTrain,...
%             'DataAugmentation',imageAugmenter,...
%             'OutputSizeMode','randcrop');
%         
%                 trainedNet = trainNetwork(datasource,lgraph,options);

%Train the network and close all training plots after training finishes.

          [trainedNet1,netinfo1] = trainNetwork(imdsTrain,lgraph_new,options1);
          close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))


% Evaluate the trained network on the validation set, calculate the predicted image labels, 
% and calculate the error rate on the validation data.

                YValidation = imdsValidation.Labels;
                 YPredicted = classify(trainedNet1,imdsValidation,'ExecutionEnvironment','gpu');
                valError1 = 1 - mean(YPredicted == YValidation);
         
%  save the network,
% validation error, and training options to disk. 


             fileName = num2str(valError) + ".mat";
        save(fileName,'trainedNet1','netinfo1','valError1','options1')
