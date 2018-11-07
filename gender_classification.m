%Dataset includes 36*36 grayscale images
%There are 2500 images seperately for men and women in training dataset
%There are 200 images seperately for men and women in testing dataset
w=36;	h=36;

%reading training dataset
fileW = dir('gender_classification\training\women\*.jpg');
for i=1:length(fileW)
    img = imread(fullfile('gender_classification\training\women',fileW(i).name));
    img_women(:,i) = reshape(img, w*h, 1);
end

fileM = dir('gender_classification\training\men\*.jpg');
for i=1:length(fileM)
    img = imread(fullfile('gender_classification\training\men',fileM(i).name));
    img_men(:,i) = reshape(img, w*h, 1);
end

img_men = im2double(img_men);
img_women = im2double(img_women);

%mean face
meanF_M = mean(img_men,2);
meanF_W = mean(img_women,2);
diff_img_M = img_men - repmat(meanF_M,1,length(fileM));		%zero-mean
diff_img_W = img_women - repmat(meanF_W,1,length(fileW));

%SVD
[u_M, d_M, v_M] = svd(diff_img_M, 0);
[u_W, d_W, v_W] = svd(diff_img_W, 0);

eigVals_M = diag(d_M); eigVals_W = diag(d_W);

for i=1:length(eigVals_W)
    energy_M(i)=sum(eigVals_M(1:i));
    energy_W(i)=sum(eigVals_W(1:i));
end

propEnergy_M = energy_M./energy_M(end);
propEnergy_W = energy_W./energy_W(end);

percentMark_W = min(find(propEnergy_W>0.90));
percentMark_M = min(find(propEnergy_M>0.90));

if percentMark_W > percentMark_M
    percentMark = percentMark_W;
else
    percentMark = percentMark_M;
end

eigenVec_M = u_M(:,1:percentMark); eigenVec_W = u_W(:,1:percentMark);


%test dataset
fileM_t = dir('gender_classification\testing\men\*.jpg');
for i=1:length(fileM_t)
    img = imread(fullfile('gender_classification\testing\men',fileM_t(i).name));
    img_men_t(:,i) = reshape(img, w*h, 1);
end

fileW_t = dir('gender_classification\testing\women\*.jpg');
for i=1:length(fileW_t)
    img = imread(fullfile('gender_classification\testing\women',fileW_t(i).name));
    img_women_t(:,i) = reshape(img, w*h, 1);
end
img_women_t = im2double(img_women_t);
img_men_t = im2double(img_men_t);

[L_M, num_M, num_W] = genderClassification(img_men_t, eigenVec_M, eigenVec_W);
s_M = num_M/length(fileM_t)		%success rate for men test dataset
[L_W, num_M, num_W] = genderClassification(img_women_t, eigenVec_M, eigenVec_W);
s_W = num_W/length(fileW_t)		%success rate for women test dataset


function [labels, n_M, n_W] = genderClassification(img, eigM, eigW)
mean_img = mean(img,2);
diff_img = img - repmat(mean_img,1,size(img,2));
w_M = eigM'*diff_img;
w_W = eigW'*diff_img;
projM = eigM*w_M;
projW = eigW*w_W;

rmse_M = sqrt(mean((diff_img-projM).^2,2));
rmse_W = sqrt(mean((diff_img-projW).^2,2));
n_M=0; n_W=0;		%number of men or women
for i=1:size(img,2)
    if rmse_M(i)<=rmse_W(i)
        labels(i) = {'M'};
        n_M = n_M+1;
    else
        labels(i) = {'W'};
        n_W = n_W+1;
    end
end
end
