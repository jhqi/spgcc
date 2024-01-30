close all;clear;clc;
disp('Entropy Rate Superpixel Segmentation Demo');

dataset='PU';
file_path=['pca_img/', dataset, '_pca.png'];
img = imread(file_path);
grey_img = double(rgb2gray(img));

n_sp=2000;

seg_res = mex_ers(grey_img, n_sp);
seg_res = int32(seg_res);
save_file_path=['seg_res/', dataset, '/', dataset, '_sp_map_', num2str(n_sp), '.mat'];
save(save_file_path,'seg_res');