% Grad-CAM active hit predictiion  visualization 

% Read the image 

img_active = imread(['E:\Predictive_DL_model\Retrain_Model_Hyperparameter_optimized\Retrained_Models_Result\' ...
    'Class_Weighted\b\Final_sel_best_models\Finalised_Models\0.027911\Pred_12_mol_SI_mols\Buparlisib.png']);


inputSize = trainedNet1.Layers(1).InputSize(1:2);

[classfn,score] = classify(trainedNet1,img_active);
imshow(img_active);
title(sprintf("%s (%.2f)", classfn, score(classfn)));
colormap jet

map = gradCAM(trainedNet1,img_active,classfn);
imshow(img_active);
hold on;
imagesc(map,'AlphaData',0.5);
colormap hot
hold off;
title(['Predicted Class: ' classfn ', Score: ' num2str(score(classfn))]);
saveas(gcf, '0.033686_model_grad_cam_SPB08257.png');
saveas(gcf, '0.033686_model_grad_cam_SPB08257.tif');
% print('output_image', '-dtiff', '-r800');
% print('grad_cam_output', '-dtiff', '-r800');


% Prediction of hit and standard inhibitor from trained model 

% Set the seed of the global random number generator to aid in the reproducibility of results.

rng('default');

vs_hits = fullfile('E:\Predictive_DL_model\Retrain_Model_Hyperparameter_optimized\Retrained_Models_Result\Class_Weighted\b\Final_sel_best_models\Finalised_Models\0.027911\Pred_12_mol_SI_mols');
imds_vs_hit_SI = imageDatastore(vs_hits, ...
    'IncludeSubfolders',true);

[VS_hit_SI_Pred,VS_hit_SI_prob] = classify(trainedNet1,imds_vs_hit_SI,'ExecutionEnvironment','gpu');

summary(VS_hit_SI_Pred)


% Grad-CAM visualization of hit and standard inhibitor 


img = imread(['E:\Predictive_DL_model\Retrain_Model_Hyperparameter_optimized\Retrained_Models_Result\' ...
    'Class_Weighted\b\Final_sel_best_models\Finalised_Models\0.027911\Pred_12_mol_SI_mols\Voxtalisib.png']);
inputSize = trainedNet1.Layers(1).InputSize(1:2);

[classfn,score] = classify(trainedNet1,img);
imshow(img);
title(sprintf("%s (%.2f)", classfn, score(classfn)));
colormap jet

map = gradCAM(trainedNet1,img,classfn);
imshow(img);
hold on;
imagesc(map,'AlphaData',0.5);
colormap hot
hold off;
%title(['Predicted Class: ' classfn ', Score: ' num2str(score(classfn))]);
saveas(gcf, 'grad_cam_Voxtalisib.png');
saveas(gcf, 'grad_cam_Voxtalisib.tif');
%print('output_image', '-dtiff', '-r800');
%print('grad_cam_output', '-dtiff', '-r800');
