function x = get_features(im, features, cell_size, cos_window,w2c)
%GET_FEATURES
%   Extracts dense features from image.  ��ͼ������ȡ�ܼ�������
%   Extracts features specified in struct FEATURES, from image IM. The features should be densely
%   sampled,in cells or intervals of CELL_SIZE.��ͼ�����ܼ���������CELL_SIZE�ĵ�Ԫ�������
%   The output has size[height in cells, width in cells, features].������д�С[��Ԫ���еĸ߶ȣ���Ԫ���еĿ�ȣ�����]  

%   To specify HOG features, set field 'hog' to true, and'hog_orientations' to the number of bins.  
%   Ҫָ��HOG���ܣ��뽫�ֶΡ�hog������Ϊtrue������hog_orientations������Ϊbin����

%   To experiment with other features simply add them to this function
%   and include any needed parameters in the FEATURES struct.
%   Ҫʵ���������ܣ�ֻ�轫����ӵ��˹��ܲ���FEATURES�ṹ�а���������κβ�����

%   To allow combinations of features, stack them with x = cat(3, x, new_feat).
	if features.hog,
		%HOG features, from Piotr's Toolbox����Piotr�Ĺ������HOG����
		x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
		x(:,:,end) = [];  
        %remove all-zeros channel ("truncation feature")
        %ɾ��ȫ��ͨ�������ضϹ��ܡ���
    end
    if features.hogcolor,
		%HOG features, from Piotr's Toolbox
		x = double(fhog(single(im) / 255, cell_size, features.hog_orientations));
		x(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
		sz = size(x);
		im_patch = imresize(im, [sz(1) sz(2)]);
		out_npca = get_feature_map(im_patch, 'gray', w2c);
		out_pca = get_feature_map(im_patch, 'cn', w2c);
% 		out_pca = reshape(temp_pca, [prod(sz), size(temp_pca, 3)]);
		x = cat(3,x,out_npca);
		x=cat(3,x,out_pca);
    end
	
	if features.gray,
		%gray-level (scalar feature)�Ҷȼ�������������
		x = double(im) / 255;
		
		x = x - mean(x(:));
	end
	
	%process with cosine window if needed�����Ҫ��ʹ�����Ҵ��ڽ��д���
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);
	end
	
end
