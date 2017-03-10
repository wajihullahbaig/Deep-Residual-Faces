function main(  )

%setup(1,struct('enableGpu',true));
lfw_path = '/media/wajih/Disk1 500 GB/Onus/RnD/DataSet/face images/lfw_home/lfw 224x224/';
resnet_path = '/media/wajih/Disk1 500 GB/Onus/Resnet Result/cacd2000/cacd2000_224x224-resnet-152/'; 
% Load the residual network 
resnet_with_stats = load([resnet_path 'net-epoch-110.mat']);
resnet = dagnn.DagNN.loadobj(resnet_with_stats.net);
resnet.mode = 'test';
%genuine_matcher(lfw_path,resnet);
imposter_matcher(lfw_path,resnet);
end

