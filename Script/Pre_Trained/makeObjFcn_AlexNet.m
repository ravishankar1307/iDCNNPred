function ObjFcn = makeObjFcn_Alexnet(imdsTrain,imdsValidation)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)

        classes = ["Active" "Inactive"];
        tbl = (countEachLabel(imdsTrain));
        totalNumber = sum(tbl.Count);
        frequency = tbl.Count / totalNumber;
        ClassWeights = 1./frequency;
        imageSize = [227,227,3];
        numClasses = numel(categories(imdsTrain.Labels));
        %initialNumFilters = round((max(imageSize)/2)/sqrt(optVars.NetworkDepth));
        numMaxPools=3;
        PoolSizeAvg = floor(max(imageSize)/(2^(numMaxPools)));
        %filterSize = 5;

        net = alexnet;%googlenet;
        %net.Layers(1);
        inputSize = net.Layers(1).InputSize;
        
%         Replace Final Layers.Extract all layers, except the last three, from the pretrained network.

            layersTransfer = net.Layers(1:end-3);

% replacing the last three layers with a fully connected layer, a softmax layer, and a classification output layer. 
% Set the fully connected layer to have the same size as the number of classes in the new data. 
% To learn faster in the new layers than in the transferred layers, increase the WeightLearnRateFactor and BiasLearnRateFactor values of the fully connected layer.

     layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer("ClassWeights",ClassWeights,"Classes",classes)];


        %figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
        %plot(lgraph)
        %ylim([0,10])


%###### Freeze Initial Layers ########
% layers = lgraph.Layers;
% connections = lgraph.Connections;
% 
% layers(1:10) = freezeWeights(layers(1:10));
% lgraph = createLgraphUsingConnections(layers,connections);
%######################################

% pixelRange = [-30 30];
% scaleRange = [0.9 1.1];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange, ...
%     'RandXScale',scaleRange, ...
%     'RandYScale',scaleRange);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),XValidation,YValidation, ...
%     'DataAugmentation',imageAugmenter);
%         augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
%         augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
%augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

        miniBatchSize = 128;
        % valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
%         valFrequency =floor(numel(imdsTrain.Labels)/miniBatchSize);
        gpuDevice(1);
        options = trainingOptions('sgdm', ...
            'InitialLearnRate',optVars.InitialLearnRate,...
            'Momentum',optVars.Momentum,...
            'ExecutionEnvironment','gpu',...
            'MiniBatchSize',miniBatchSize, ...
            'L2Regularization',optVars.L2Regularization,...
            'MaxEpochs',35, ...
            'Shuffle','every-epoch', ...
            'ValidationData',imdsValidation, ...
            'ValidationFrequency',30, ...
            'Verbose',false, ...
            'Plots','none');

        %  'ExecutionEnvironment','multi-gpu',...
%            'Plots','training-progress');
       %     'Plots','none');
        
        %    'Plots','training-progress');

        [trainedNet,~] = trainNetwork(imdsTrain,layers,options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))
                
        YValidation = imdsValidation.Labels;
        YPredicted = classify(trainedNet,imdsValidation,"ExecutionEnvironment","gpu");
        valError = 1 - mean(YPredicted == YValidation);
        
        fileName = num2str(valError) + ".mat";
        save(fileName,'trainedNet','valError','options')
        cons = [];
        
%         options = trainingOptions('sgdm',...
%             'InitialLearnRate',optVars.InitialLearnRate,...
%             'Momentum',optVars.Momentum,...
%             'ExecutionEnvironment','multi-gpu',...
%             'MaxEpochs',10, ...
%             'LearnRateSchedule','piecewise',...
%             'LearnRateDropPeriod',35,...
%             'LearnRateDropFactor',0.1,...
%             'MiniBatchSize',miniBatchSize,...
%             'L2Regularization',optVars.L2Regularization,...
%             'Shuffle','every-epoch',...
%             'Verbose',false,...
%             'Plots','none',...
%             'ValidationData',{XValidation,YValidation},...
%             'ValidationPatience',Inf,...
%             'ValidationFrequency',validationFrequency);
  %'Plots','none',...     
  %'MaxEpochs',100,...
  
       %  'Plots','training-progress',...
        
%         imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-5,5], ...
%     'RandXTranslation',[-3 3], ...
%     'RandYTranslation',[-3 3]);
%  
%         datasource = augmentedImageDatastore(imageSize,XTrain,YTrain,...
%             'DataAugmentation',imageAugmenter,...
%             'OutputSizeMode','randcrop');
        



%                 trainedNet = trainNetwork(datasource,lgraph,options);
%            trainedNet = trainNetwork(XTrain,YTrain,lgraph,options);
%         close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))
%         
%                YPredicted = classify(trainedNet,XValidation);
%         valError = 1 - mean(YPredicted == YValidation);
%         
%              fileName = num2str(valError) + ".mat";
%         save(fileName,'trainedNet','valError','options')
%         cons = [];
        
    end
end
 
