function out = get_feature_map(im_patch, features, w2c)
% 提取图像特征，在get_subwindow函数中调用，用来计算PCA特征和non-PCA特征
% out = get_feature_map(im_patch, features, w2c)
% 从图像块中提取指定特征，w2c是CN矩阵
% Extracts the given features from the image patch. 
% w2c is the Color Names matrix, if used.

if nargin < 3
    w2c = [];
end

% the names of the features that can be used
valid_features = {'gray', 'cn'};

% the dimension of the valid features
feature_levels = [1 10]';

num_valid_features = length(valid_features);
used_features = false(num_valid_features, 1); %false用来生成logical 类型矩阵，里面参数为维度

% get the used features
for i = 1:num_valid_features
    used_features(i) = any(strcmpi(valid_features{i}, features)); %比较字符串，不区分大小  any函数：检测矩阵中是否有非零元素，如果有，则返回1，否则，返回0
end

% total number of used feature levels
num_feature_levels = sum(feature_levels .* used_features);

level = 0;

% If grayscale image
if size(im_patch, 3) == 1
    % Features that are available for grayscale sequances
    
    % Grayscale values (image intensity)
    out = single(im_patch)/255 - 0.5;
else
    % Features that are available for color sequances
    
    % allocate space (for speed)
    out = zeros(size(im_patch, 1), size(im_patch, 2), num_feature_levels, 'single');
    
    % Grayscale values (image intensity)
    if used_features(1)
        out(:,:,level+1) = single(rgb2gray(im_patch))/255 - 0.5; %灰度值作为第一层特征
        level = level + feature_levels(1);
    end
    
    % Color Names
    if used_features(2)
        if isempty(w2c)
            % load the RGB to color name matrix if not in input
            temp = load('w2crs');
            w2c = temp.w2crs;
        end
        
        % extract color descriptor
        out(:,:,level+(1:10)) = im2c(single(im_patch), w2c, -2);
        level = level + feature_levels(2); 
    end
end