% initialize the variables
alphabet=char(65:90);
hangle = ['ぁ', 'い', 'ぇ', 'ぉ','け','げ','さ','し','じ','ず','ぜ','せ','ぜ','そ','ぞ','た','ち','っ','づ','で','に','ぬ','ば','ぱ','び','だ','ぢ','つ','て','ひ','な','は'];
trainingFeatures=[];
trainlabel=[];

%unzip('HanguelData.zip');
imds = imageDatastore('hangul', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);
net = googlenet;
analyzeNetwork(net);
inputSize = net.Layers(1).InputSize;
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]
numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);
save('net2','net');
[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels)
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));  
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

% save the label, features

% using trainingSet, execute the svmtraining in cost of 10
% generate the svm model
%svmstruct=svmtrain(trainlabel,trainingFeatures,'-c 10');
%save('svmstruct','svmstruct');


% % test on the svm model
% alphabet=char(65:90);
% table=zeros(25,25);
% for d=1:25
%     for i=1:30
%         if(d==10)
%             continue;
%         end
%         combinedStr=strcat('hangle/',alphabet(d),'/',alphabet(d),'_',num2str(i),'.png');
%         img=imread(combinedStr);
%         
%         % resize image to 100*100 
%         img=imresize(img,[100,100]);
%         img=double(img);
%     
%         % extract HOG features from testSet
%         testset=double(extractHOGFeatures(img,'CellSize',[4 4]));
%         testlabel=d;
%         
%         % check whether the model is correct or not
%         predict=svmpredict(testlabel,testset,svmstruct);
%         if(testlabel~=predict)
%             table(testlabel,predict)=table(testlabel,predict)+1;
%         end
%     end
% end