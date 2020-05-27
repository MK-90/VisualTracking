function z = feature_projection(x_npca, x_pca, projection_matrix, cos_window)

% z = feature_projection(x_npca, x_pca, projection_matrix, cos_window)
%
% Calculates the compressed feature map by mapping the PCA features with
% the projection matrix and concatinates this with the non-PCA features.
% The feature map is then windowed.

%用来特征降维和加窗
%如果没有PCA特征，只有用non-PCA特征
%如果有PCA特征的话：
%        将PCA特征使用投影矩阵投影，并且调整大小
%        然后再看，如果有PCA特征但是没有non-PCA特征，只有用PCA特征
%                         如果两种特征都有，将两者叠加起来
%最后给投影后的特征加余弦窗


if isempty(x_pca)
    % if no PCA-features exist, only use non-PCA
    z = x_npca;
else
    % get dimensions
    [height, width] = size(cos_window);
    [num_pca_in, num_pca_out] = size(projection_matrix);
    
    % project the PCA-features using the projection matrix and reshape
    % to a window 使用投影矩阵计算PCA特征，并且调整大小
    x_proj_pca = reshape(x_pca * projection_matrix, [height, width, num_pca_out]);
    
    % concatinate the feature windows
    if isempty(x_npca)
        z = x_proj_pca;
    else
        z = cat(3, x_npca, x_proj_pca);
    end
end

% do the windowing of the output 使用余弦窗
z = bsxfun(@times, cos_window, z);
end