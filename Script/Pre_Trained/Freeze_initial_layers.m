%Freeze Initial Layers
% "freeze" the weights of earlier layers in the network by setting the learning rates in those layers to zero. 
% During training, trainNetwork does not update the parameters of the frozen layers.Freezing the weights of many initial layers can significantly speed up network training and also prevent those layers from overfitting.

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:63) = freezeWeights(layers(1:63));
lgraph = createLgraphUsingConnections(layers,connections);