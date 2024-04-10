%% Compute Activations for Several Layers
earlyLayerName = "MaxPool_1";
finalConvLayerName = "conv_5";
softmaxLayerName = "softmax";
pool1Activations = activations(trainedNet1,...
    imdsTest,earlyLayerName,"OutputAs","rows");
finalConvActivations = activations(trainedNet1,...
    imdsTest,finalConvLayerName,"OutputAs","rows");
softmaxActivations = activations(trainedNet1,...
    imdsTest,softmaxLayerName,"OutputAs","rows");

% Ambiguity of Classifications

[R,RI] = maxk(softmaxActivations,2,2);
ambiguity = R(:,2)./R(:,1);

[ambiguity,ambiguityIdx] = sort(ambiguity,"descend");


classList = unique(imdsTest.Labels);
top10Idx = ambiguityIdx(1:10);
top10Ambiguity = ambiguity(1:10);
mostLikely = classList(RI(ambiguityIdx,1));
secondLikely = classList(RI(ambiguityIdx,2));
table(top10Idx,top10Ambiguity,mostLikely(1:10),secondLikely(1:10),imdsTest.Labels(ambiguityIdx(1:10)),...
    'VariableNames',["Image #","Ambiguity","Likeliest","Second","True Class"])


 v = 66;
figure();
imshow(imdsTest.Files{v});
title(sprintf("Observation: %i\n" + ...
    "Actual: %s. Predicted: %s", v, ...
    string(imdsTest.Labels(v)), string(TestPred(v))), ...
    'Interpreter', 'none');


%% Compute 2-D Representations of Data Using t-SNE

rng default
pool1tsne = tsne(pool1Activations);
finalConvtsne = tsne(finalConvActivations);
softmaxtsne = tsne(softmaxActivations);

doLegend = 'off';
markerSize = 7;
figure;
% Increase title font size for all subplots
titleFontSize = 20; % Adjust as needed


subplot(1,3,1);
gscatter(pool1tsne(:,1),pool1tsne(:,2),imdsTest.Labels, ...
    [],'.',markerSize,doLegend);
title("Max pooling activations");

subplot(1,3,2);
gscatter(finalConvtsne(:,1),finalConvtsne(:,2),imdsTest.Labels, ...
    [],'.',markerSize,doLegend);
title("Final conv activations");

subplot(1,3,3);
gscatter(softmaxtsne(:,1),softmaxtsne(:,2),imdsTest.Labels, ...
    [],'.',markerSize,doLegend);
title("Softmax activations");

% Increase legend font size
legendFontSize = 20; % Adjust as needed

numClasses = length(classList);
colors = lines(numClasses);
h = figure;
gscatter(softmaxtsne(:,1),softmaxtsne(:,2),imdsTest.Labels,colors);

l = legend;
l.Interpreter = "none";
l.Location = "bestoutside";

print(figure(1), 't-SNE_activations_trained_model.png', '-dpng', '-r800')

print(figure(1), 't-SNE_activations_trained_model.tif', '-dtiff', '-r800')

print(figure(h), 't-SNE_classes_plot_trained_model.png', '-dpng', '-r800')

print(figure(h), 't-SNE_classes_plot_trained_model.tif', '-dtiff', '-r800')