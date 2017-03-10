function [net, info] = res_cacd2000_224x224(m, varargin)
% res_cifar(20, 'modelType', 'resnet', 'reLUafterSum', false,...
% 'expDir', 'data/exp/cifar-resNOrelu-20', 'gpus', [2])
%setup;
opts.modelType = 'plain' ;
opts.preActivation = false;
opts.reLUafterSum = false;
opts.shortcutBN = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.preActivation ,
    opts.expDir = fullfile('exp', ...
        sprintf('cacd2000_224x224-%s-%d', opts.modelType,m)) ;
else
    opts.expDir = fullfile('exp', ...
        sprintf('cacd2000_224x224-resnet-Pre-%d',m)) ;
end
opts.dataDir = fullfile('/media/wajih/Disk1 500 GB/Onus/RnD/DataSet/','face images') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.whitenData = true;
opts.contrastNormalization = true;
opts.meanType = 'image'; % 'pixel' | 'image'
opts.border = [4 4 4 4]; % tblr
opts.gpus = [];
opts.checkpointFn = [];
opts = vl_argparse(opts, varargin) ;

if numel(opts.border)~=4,
    assert(numel(opts.border)==1);
    opts.border = ones(1,4) * opts.border;
end

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

if opts.preActivation ,
    net = res_cacd2000_224x224_preactivation_init(m) ;
else
    net = res_cacd2000_224x224_init(m, 'networkType', opts.modelType, ...
        'reLUafterSum', opts.reLUafterSum) ;
end

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
    if ~strcmpi(imdb.meta.meanType, opts.meanType) ...
            || xor(imdb.meta.whitenData, opts.whitenData) ...
            || xor(imdb.meta.contrastNormalization, opts.contrastNormalization);
        clear imdb;
    end
end
if ~exist('imdb', 'var'),
    imdb = getcacd2000_224x224(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb','-v7.3') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;
net.meta.dataMean = imdb.meta.dataMean;
augData = zeros(size(imdb.images.data) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imdb.images.data);
augData(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data;
imdb.images.augData = augData;


% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train_dag_check;

rng('default');
[net, info] = trainfn(net, imdb, getBatch(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    'gpus', opts.gpus, ...
    'val', find(imdb.images.set == 3), ...
    'derOutputs', {'loss', 1}, ...
    'checkpointFn', opts.checkpointFn) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
if imdb.images.set(batch(1))==1,  % training
    sz0 = size(imdb.images.augData);
    sz = size(imdb.images.data);
    loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
    images = imdb.images.augData(loc(1):loc(1)+sz(1)-1, ...
        loc(2):loc(2)+sz(2)-1,:, batch);
    if rand > 0.5, images=fliplr(images) ; end
else                              % validating / testing
    images = imdb.images.data(:,:,:,batch);
end
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'image', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getcacd2000_224x224(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
downloadPath = strcat(opts.dataDir,'/cacd2000_224x224_300x15.mat');

if ~exist(strcat(downloadPath), 'file')
    url = 'https://github.com/wajihullahbaig/DeepNeuralNetworks/tree/master/data/mnist_uint8.mat' ;
    fprintf('downloading %s\n', url) ;
    websave(downloadPath,url)
    
end
% Create trainging/testing sets
% Make sure dataset size is a whole number, so that every batch loaded 
% is of equal size
dataset = load(downloadPath);
dataset = dataset.classificationData;
totalClasses = size(dataset,2);
traintestRatio = 0.8;
batchSize = 100; 
trainSetSize = round(totalClasses * dataset(1).fileCount * traintestRatio);
testSetSize = round(totalClasses * dataset(1).fileCount * (1-traintestRatio));
% trainsize/batchsize and testsize/batchsize must be whole 
% numbers
file_set = uint8([ones(1, trainSetSize/batchSize), 3*ones(1,testSetSize/batchSize)]);
data = cell(1, totalClasses * dataset(1).fileCount/batchSize);
labels = cell(1, totalClasses * dataset(1).fileCount/batchSize);
sets = cell(1, totalClasses * dataset(1).fileCount/batchSize);

fullTrainingData = ones(224,224,3,trainSetSize);
fullTrainingLabels = zeros(trainSetSize,1);
fullTestingData = zeros(224,224,3,testSetSize);
trueLabels = zeros(testSetSize,1);

count1 = 0;
count2 = 0;
trainImagesCount = floor(dataset(1).fileCount*traintestRatio);
for i = 1:totalClasses
     for j = 1: trainImagesCount
     img = double(imread(strcat(opts.dataDir,'/CACD2000_224x224/', dataset(i).fileList{j})))./255.0;
     fullTrainingData(:,:,:,i+count1) = img;
     fullTrainingLabels(i+count1,1) = i;
     count1 = count1+1;
     end
     for j= trainImagesCount+1:dataset(1).fileCount
     img = double(imread(strcat(opts.dataDir,'/CACD2000_224x224/', dataset(i).fileList{j})))./255.0;
     fullTestingData (:,:,:,i+count2) = img;
     trueLabels(i+count2,1) = i;
     count2 =  count2+1;
     end
     count1 = count1-1;
     count2 = count2-1;
 end
 

for i= 1:trainSetSize/batchSize  
  data{i} = fullTrainingData(:,:,:,(i - 1) * batchSize + 1 : i * batchSize);  
  batch = fullTrainingLabels((i - 1) * batchSize + 1 : i * batchSize,1);
  labels{i} = batch'; % Index from 1
  sets{i} = repmat(file_set(i), size(labels{i}));
end

index = trainSetSize/batchSize;
for i= 1:testSetSize/batchSize  
    
  data{index+i} = fullTestingData(:,:,:,(i - 1) * batchSize + 1 : i * batchSize);  
  batch = trueLabels((i - 1) * batchSize + 1 : i * batchSize,1);
  labels{index+i} = batch'; % Index from 1
  sets{index+i} = repmat(file_set(index+i), size(labels{index+i}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean
dataMean = mean(data(:,:,:,set == 1), 4);
if strcmpi(opts.meanType, 'pixel'),
    dataMean = mean(mean(dataMean, 1), 2);
elseif ~strcmpi(opts.meanType, 'image'),
    error('Unknown option: %s', opts.meanType);
end
data = bsxfun(@minus, data, dataMean);

% % normalize by image mean and std as suggested in `An Analysis of
% % Single-Layer Networks in Unsupervised Feature Learning` Adam
% % Coates, Honglak Lee, Andrew Y. Ng
%
% if opts.contrastNormalization
%   z = reshape(data,[],70000) ;
%   z = bsxfun(@minus, z, mean(z,1)) ;
%   n = std(z,0,1) ;
%   z = bsxfun(@times, z, mean(n) ./ max(n, 36)) ;
%   data = reshape(z, 28, 28, 3,[]) ; % Later lets try 28,28,1 reshape
% end
%
% if opts.whitenData
%   z = reshape(data,[],70000) ;
%   W = z(:,set == 1)*z(:,set == 1)'/70000 ;
%   [V,D] = eig(W) ;
%   % the scale is selected to approximately preserve the norm of W
%   d2 = diag(D) ;
%   en = sqrt(mean(d2)) ;
%   z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
%   data = reshape(z, 28, 28,3, []) ;
% end

for i= 1:totalClasses
    clNames.label_names{i,1} = num2str(i);
end
imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.meta.dataMean = dataMean;
imdb.meta.meanType = opts.meanType;
imdb.meta.whitenData = opts.whitenData;
imdb.meta.contrastNormalization = opts.contrastNormalization;
