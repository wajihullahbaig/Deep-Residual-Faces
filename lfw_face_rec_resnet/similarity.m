function [s] = similarity(v1,v2)
v1 = v1-min(v1(:));
v1 = v1./max(v1(:));
v2 = v2-min(v2(:));
v2 = v2./max(v2(:));
normV1 = norm(v1);
normV2 = norm(v2);
s = 1-norm(normV1-normV2)/(normV1+normV2);
%s = 1- dot(v1,v2)/(normV1)*(normV2);
return ;