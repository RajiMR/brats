%fresh start
clc
clear all
close all

gpuDevice(2);


% brats training and validation dataset locations
trImLoc = fullfile('/rsrch1/ip/rmuthusivarajan/imaging/BraTS/preprocessedDataset/imagesTr');
valImLoc = fullfile('/rsrch1/ip/rmuthusivarajan/imaging/BraTS/preprocessedDataset/imagesVal');

TrLblLoc = fullfile('/rsrch1/ip/rmuthusivarajan/imaging/BraTS/preprocessedDataset/labelsTr');
valLblLoc = fullfile('/rsrch1/ip/rmuthusivarajan/imaging/tmp/BraTS/preprocessedDataset/labelsVal');

% training set: make a loop to read in all .mat files and convert to slices
trainImset = dir(trImLoc);
trainLblset = dir(TrLblLoc);
numFiles = length(trainImset)-2;
trainImTable = struct2table(trainImset);
trainLblTable = struct2table(trainLblset);

% declare cell arrays to store slice data
trImCell = {};
trLblCell = {};

numIm = 1;
numLbl = 1;


for id = 1:400
    voldestcell = fullfile(trImLoc, trainImTable{id+2,"name"});
    voldest = cell2mat(voldestcell);
    imLoad = load(voldest);
    imVol = imLoad.cropVol;
    inCell = images4d2store(imVol);
    
    for im = 1:numel(inCell)
        trImCell{numIm} = inCell{im};
        numIm = numIm + 1;
    end
    
    lbldestcell = fullfile(TrLblLoc, trainLblTable{id+2, "name"});
    lbldest = cell2mat(lbldestcell);
    lblLoad = load(lbldest);
    lblVol = lblLoad.cropLabel;
    inCell = labels4d2store(lblVol);
    
    for im = 1:numel(inCell)
        trLblCell{numLbl} = inCell{im};
        numLbl = numLbl + 1;
    end
    
end

% concatenate along the 4th dimension to make arrays
TrImCat = cat(4, trImCell{:});
TrLblCat = cat(4, trLblCell{:});

% make training augmented ds
trAugimds = augmentedImageDatastore([128 128 4], TrImCat, TrLblCat);


% validation set: make a loop to read in all.mat files and convert to slices
valImset = dir(valImLoc);
valLblset = dir(valLblLoc);
valImTable = struct2table(valImset);
valLblTable = struct2table(valLblset);
numFiles = length(valImset)-2;

% declare cell arrays to store slices
valImCell = {};
valLblCell = {};

numIm = 1;
numLbl = 1;

for id = 1:numFiles
    voldestcell = fullfile(valImLoc, valImTable{id+2, "name"});
    voldest = cell2mat(voldestcell);
    imLoad = load(voldest);
    imVol = imLoad.cropVol;
    inCell = images4d2store(imVol);
    
    for im = 1:numel(inCell)
        valImCell{numIm} = inCell{im};
        numIm = numIm + 1;
    end
    
    lbldestcell = fullfile(valLblLoc, valLblTable{id+2, "name"});
    lbldest = cell2mat(lbldestcell);
    lblLoad = load(lbldest);
    lblVol = lblLoad.cropLabel;
    inCell=labels4d2store(lblVol);
    
    for im = 1:numel(inCell)
        valLblCell{numLbl} = inCell{im};
        numLbl = numLbl + 1;
    end
    
end

% concatenate validation arrays
valImCat = cat(4, valImCell{:});
valLblCat = cat(4, valLblCell{:});

% validation augmented ds
valAugimds = augmentedImageDatastore([128 128 4], valImCat, valLblCat);

%% load network
%specify the n as the number of channels
n = 4;
% Create Layer Graph
% Create the layer graph variable to contain the network layers.

lgraph = layerGraph();
% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    imageInputLayer([128 128 n],"Name","input","Normalization","none")
    convolution2dLayer([3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    reluLayer("Name","relu_Module1_Level1")
    convolution2dLayer([3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module2_Level1")
    reluLayer("Name","relu_Module2_Level1")
    convolution2dLayer([3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    batchNormalizationLayer("Name","BN_Module3_Level1")
    reluLayer("Name","relu_Module3_Level1")
    convolution2dLayer([3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")
    convolution2dLayer([3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")
    transposedConv2dLayer([2 2],512,"Name","transConv_Module4","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat3")
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level1")
    convolution2dLayer([3 3],256,"Name","conv_Module5_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module5_Level2")
    transposedConv2dLayer([2 2],256,"Name","transConv_Module5","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat2")
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level1")
    convolution2dLayer([3 3],128,"Name","conv_Module6_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module6_Level2")
    transposedConv2dLayer([2 2],128,"Name","transConv_Module6","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(3,2,"Name","concat1")
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level1","Padding","same")
    reluLayer("Name","relu_Module7_Level1")
    convolution2dLayer([3 3],64,"Name","conv_Module7_Level2","Padding","same")
    reluLayer("Name","relu_Module7_Level2")
    convolution2dLayer([1 1],2,"Name","ConvLast_Module7")
    softmaxLayer("Name","softmax")
    dicePixelClassificationLayer('Name', 'output')];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"relu_Module1_Level2","maxpool_Module1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","concat1/in2");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","maxpool_Module2");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","concat2/in2");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","maxpool_Module3");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","concat3/in2");
lgraph = connectLayers(lgraph,"transConv_Module4","concat3/in1");
lgraph = connectLayers(lgraph,"transConv_Module5","concat2/in1");
lgraph = connectLayers(lgraph,"transConv_Module6","concat1/in1");
% Plot Layers

%plot(lgraph);

%% train the model on the training set for each fold in the k-fold
% Need to Train the network using training and validation data

options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',valAugimds, ...
    'ValidationFrequency',400, ...
    'ValidationPatience',3, ...
    'Plots','training-progress', ...
    'Verbose',false);

doTraining = true;
if doTraining 
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(trAugimds,lgraph,options);
    save(['BraTStrained2DUNet-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.mat'],'net');
    infotable = struct2table(info);
    writetable(infotable, ['BraTS2DUNetinfo-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.txt']);
end
