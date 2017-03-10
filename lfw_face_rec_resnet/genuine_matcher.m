function genuine_matcher( lfw_path , resnet)
directories = dir(fullfile(lfw_path));

for i = 1:numel(directories)
    directory_name = directories(i).name;
    if strcmp(directory_name,'.') == 1 || strcmp(directory_name,'..') == 1
        continue;
    end
    for j = 1:8
        fileName1 = [lfw_path directory_name '/' directory_name '_000' num2str(j) '.jpg'];
        img1 = imread(fileName1);
        img1 = single(img1./255);
        img1 = imresize(img1,resnet.meta.normalization.imageSize(1:2));
        img1 = bsxfun(@minus,img1,resnet.meta.dataMean);
        resnet.eval({'image',img1});
        feature1 = resnet.vars(resnet.getVarIndex('fc201')).value;
        for p=j+1:8
            fileName2 = [lfw_path directory_name '/' directory_name '_000' num2str(p) '.jpg'];
            img2 = imread(fileName2);
            img2 = single(img2./255);
            img2 = imresize(img2,resnet.meta.normalization.imageSize(1:2));
            img2 = bsxfun(@minus,img2,resnet.meta.dataMean);
            resnet.eval({'image',img2});
            feature2 = resnet.vars(resnet.getVarIndex('fc201')).value;
            similarity(squeeze(feature1),squeeze(feature2))
           
        end
    end
end
end
