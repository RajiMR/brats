% Clear workspace
clearvars; close all; clc;

gpuDevice(1)

destinationOut = '';

%% create datastores for processed labels and images
% Images Datapath % define reader
procVolReader = @(x) niftiread(x);
procVolLoc = fullfile(destination,'imgNormalized');
procVolDs = imageDatastore(procVolLoc, ...
    'FileExtensions','.nii','LabelSource','foldernames','ReadFcn',procVolReader);

procLblReader =  @(x) uint8(niftiread(x));
procLblLoc = fullfile(destination,'lblOrient');
classNames = ["background", "tumour", "nonenhancingtumor","enhancingtumour"];
pixelLabelID = [0 1 2 3]; 
procLblDs = pixelLabelDatastore(procLblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.nii','ReadFcn',procLblReader);

%%Load test indices 
s1 = load('idxHold.mat');
c1 = struct2cell(s1);
idxHold = cat(1,c1{:});

s2 = load('idxTrain.mat');
c2 = struct2cell(s2);
idxTrain = cat(1,c2{:});

s3 = load('idxVal.mat');
c3 = struct2cell(s3);
idxVal = cat(1,c3{:});

for idxFold = 1:5

    disp(['Processing K-fold-' num2str(idxFold)]);

    imdsHold = subset(procVolDs,idxHold{1,idxFold}); %imds for holdout partition
    pxdsHold = subset(procLblDs,idxHold{1,idxFold}); %imds for holdout partition

    imdsTrain = subset(imdsHold,idxTrain{1,idxFold}); %training imagedatastore
    pxdsTrain = subset(pxdsHold,idxTrain{1,idxFold}); %training pixelimagedatastore

    imdsVal = subset(imdsHold,idxVal{1,idxFold}); %training imagedatastore
    pxdsVal = subset(pxdsHold,idxVal{1,idxFold}); %training pixelimagedatastore

    %Need Random Patch Extraction on training and validation Data
    patchSize = [64 64 64];
    patchPerImage = 16;
    miniBatchSize = 8;

%training patch datastore
    trPatchDs = randomPatchExtractionDatastore(imdsTrain,pxdsTrain,patchSize, ...
    'PatchesPerImage',patchPerImage);
    trPatchDs.MiniBatchSize = miniBatchSize;

%validation patch datastore
    valPatchDs = randomPatchExtractionDatastore(imdsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
    valPatchDs.MiniBatchSize = miniBatchSize;

%%3D densenet 
%% Create Layer Graph
% Create the layer graph variable to contain the network layers.
%define n as number of channels
n = 4;
lgraph = layerGraph();

%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.
% Helper function for densenet3d upsample3dLayer.m

tempLayers = [
    image3dInputLayer([64 64 64 n],"Name","input","Normalization","none")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module1_Level2")
    convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","concat_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module2_Level1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module2_Level2")
    convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","concat_2");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module3_Level1")
    convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module3_Level2")
    convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = concatenationLayer(4,2,"Name","concat_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module4_Level1")
    convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module4_Level2")
    convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_4")
    upsample3dLayer([2 2 2],512,"Name","upsample_Module4","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat3")
    batchNormalizationLayer("Name","BN_Module5_Level1")
    convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module5_Level2")
    convolution3dLayer([3 3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_5")
    upsample3dLayer([2 2 2],256,"Name","upsample_Module5","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat2")
    batchNormalizationLayer("Name","BN_Module6_Level1")
    convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module6_Level2")
    convolution3dLayer([3 3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_6")
    upsample3dLayer([2 2 2],128,"Name","upsample_Module6","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat1")
    batchNormalizationLayer("Name","BN_Module7_Level1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module7_Level2")
    convolution3dLayer([3 3 3],64,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat")
    batchNormalizationLayer("Name","BN_Module7_Level3")
    convolution3dLayer([1 1 1],4,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassification3dLayer("output")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"relu_Module1_Level1","BN_Module1_Level2");
lgraph = connectLayers(lgraph,"relu_Module1_Level1","concat_1/in1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat_1/in2");
lgraph = connectLayers(lgraph,"concat_1","maxpool_Module1");
lgraph = connectLayers(lgraph,"concat_1","concat1/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","BN_Module2_Level2");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","concat_2/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat_2/in2");
lgraph = connectLayers(lgraph,"concat_2","maxpool_Module2");
lgraph = connectLayers(lgraph,"concat_2","concat2/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","BN_Module3_Level2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","concat_3/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat_3/in2");
lgraph = connectLayers(lgraph,"concat_3","maxpool_Module3");
lgraph = connectLayers(lgraph,"concat_3","concat3/in1");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","BN_Module4_Level2");
lgraph = connectLayers(lgraph,"relu_Module4_Level1","concat_4/in1");
lgraph = connectLayers(lgraph,"relu_Module4_Level2","concat_4/in2");
lgraph = connectLayers(lgraph,"upsample_Module4","concat3/in2");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","BN_Module5_Level2");
lgraph = connectLayers(lgraph,"relu_Module5_Level1","concat_5/in1");
lgraph = connectLayers(lgraph,"relu_Module5_Level2","concat_5/in2");
lgraph = connectLayers(lgraph,"upsample_Module5","concat2/in2");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","BN_Module6_Level2");
lgraph = connectLayers(lgraph,"relu_Module6_Level1","concat_6/in1");
lgraph = connectLayers(lgraph,"relu_Module6_Level2","concat_6/in2");
lgraph = connectLayers(lgraph,"upsample_Module6","concat1/in2");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","BN_Module7_Level2");
lgraph = connectLayers(lgraph,"relu_Module7_Level1","concat/in2");
lgraph = connectLayers(lgraph,"relu_Module7_Level2","concat/in1");

%% Plot Layers

%%plot(lgraph);

%% do the training %%
options = trainingOptions('sgdm', ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',valPatchDs, ...
    'ValidationFrequency',200, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);

    [net,info] = trainNetwork(trPatchDs,lgraph,options);
    save(['fold_' num2str(idxFold) '_trainedNet.mat'],'net');
    save(['fold_' num2str(idxFold) '_info.mat'],'info');

    handle = findall(groot, 'Type', 'Figure');
    trainingProgress = ['fold_' num2str(idxFold) '_trainingProgress.png'];
    exportapp(handle(1),trainingProgress);
end
