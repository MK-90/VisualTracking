function [positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization)

temp = load('w2crs');
w2c = temp.w2crs;   
	resize_image = (sqrt(prod(target_sz)) >= 100);%prod将A矩阵不同维的元素的乘积返回到矩阵B。 
    if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
    end %y = floor(x) 函数将x中元素取整，值y为不大于本身的最小整数
    %resize_image指示是否需要对原图进行缩放，因为跟踪区域过大时，算法对图像进行了下采样，
    %有可能跟踪中所用的图像比原图要小，因此需要此参数进行指示。

	%window size, taking padding into account 窗口大小，把padding考虑进去
	window_sz = floor(target_sz * ( 1+padding));
    %目标框向外扩展1.5倍作为window_sz
    %后面所有的处理都用window_sz,即包含目标和背景
	
% 	we could choose a size that is a power of two, for better FFT performance
% 	我们可以选择一个大小为2的幂，用于更好的FFT表现。 
%   in practice it is slower, due to the larger window size.
%   在实践中，由于窗口尺寸较大，速度较慢。% 	window_sz = 2 .^ nextpow2(window_sz);
%   create regression labels, gaussian shaped, with a bandwidth proportional to target size
%   创建回归标签，高斯形，带宽与目标尺寸成比例
	
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
%     fprintf('the  value of output_sigma is%6.2f\n',output_sigma)
    %cellsize为每一个细胞中像素的数量（HOG），若不用HOG则为1
    % output_sigma_factor 是 回归目标相对于目标大小的空间带宽
    % B = prod(A)将A矩阵不同维的元素的乘积返回到矩阵B。
    
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size))); % yf是频域上的回归值
    %cell_size是每个单元格的像素数（如果使用原始像素，则必须为1）
    
    % 至此生成了回归值，使用yf的尺寸来生成相应的余弦窗
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';
    % 余弦窗 减少图像边缘不连续性对变换结果的影响
    % hann(size(yf,1)返回行数，hann(size(yf,2))返回列数

	if show_visualization,  %create video interface 创建视频界面
		update_visualization = show_video(img_files, video_path, resize_image);
    end
	
	%note: variables ending with 'f' are in the Fourier domain.   %变量在傅立叶域中以'f'结尾。
	time = 0;  %to calculate FPS
	positions = zeros(numel(img_files), 2);  %to calculate precision
    % POSITIONS是目标位置随时间推移的Nx2矩阵，用来存放每一帧计算出的位置。
    % numel(img_files)视频的帧数
    
	for frame = 1:numel(img_files),
		%load image 加载图片
		im = imread([video_path img_files{frame}]);%读取一帧图像
		if size(im,3) > 1,
			im = rgb2gray(im);%大于3通道的，即彩色照片转为灰度照片
		end
		if resize_image,
			im = imresize(im, 0.5);%该函数用于对图像做缩放处理，如目标过大。
		end

		tic()

		if frame > 1,
            %通过上一帧的结果获取子窗口
            %obtain a subwindow for detection at the position from last
            %frame, and convert to Fourier domain (its size is unchanged)
            %在最后一帧的位置获取检测子窗口，并转换为傅立叶域（其大小不变）
			patch = get_subwindow(im, pos, window_sz);%图像块
			zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
			%对提取到的特征进行傅里叶变换，得到zf，zf是测试样本
            
			%calculate response of the classifier at all shifts 
            %计算分类器对于所有循环移位后的样本的响应
			switch kernel.type
			case 'gaussian',
                % 通过对测试样本的核变换后得到kzf
				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);%计算zf与模型的核相关矩阵	
			case 'polynomial',%zf就是特征的傅里叶变换，
				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);%model_xf
			case 'linear',
				kzf = linear_correlation(zf, model_xf);
            end
            %计算响应值  %equation for fast detection 快速检测方程
			response = real(ifft2(model_alphaf .* kzf)); % .*元素点乘 real返回实部（复数）model_alphaf 模型
            %disp(response);
           
			%target location is at the maximum response. we must take into account the fact that,
			% if the target doesn't move, the peak will appear at the top-left corner,
			% not at the center (this is discussed in the paper).the responses wrap around cyclically.
			% 目标位置处于最大响应。 我们必须考虑到，如果目标不移动，峰值将出现在左上角，
            %而不是在中心（这在本文中讨论）。这些响应周期性地绕过。
            
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
            %根据极大值在矩阵中的位置，求出当前帧的目标中心的预测值即找到响应最大的位置
             %find(X,k)，返回X中第k个非零元素的行列位置，[vert_delta, horiz_delta]行列位置
             
            %在核相关滤波的目标跟踪中，最大响应值越大，其周围的扩散度越小，该最大响应
            %值对应的位置为目标中心位置的置信度越高。
           
			if vert_delta > size(zf,1) / 2, 
                %wrap around to negative half-space of vertical axis
                %缠绕到垂直轴的负半空间
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis 水平轴相同
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];%pos位置，更新出目标的新位置
        end %通过快速检测方程得到目标，然后通过目标得到pos的中心位置
        
        
%******************************%模型更新过程
		%obtain a subwindow for training at newly estimated target position
        %获取目标位置和窗口大小
		patch = get_subwindow(im, pos, window_sz);
		xf = fft2(get_features(patch, features, cell_size, cos_window,w2c));%从图像中提取密集特征转到傅里叶域，xf为输入的训练样本
        % get_features从图像中提取密集特征。，进行傅里叶变换
        
		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
        %核岭回归，计算α（权值）（傅立叶域）
		switch kernel.type%选择用什么核
		case 'gaussian',%用高斯核
			kf = gaussian_correlation(xf, xf, kernel.sigma);%kernel.sigma高斯内核带宽	
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
        end
        
		alphaf = yf ./ (kf + lambda);   %equation for fast training快速训练方程式 ./元素间的除
        %lambda正规化
        
		if frame == 1,  %first frame, train with a single image第一帧，用单一图像训练
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model后续帧，插值模型
			%后帧的更新,使用本帧和前一帧中结果的加权
            %对前后帧结果进行了线性插值，为了让他长记性，不至于模型剧烈变化。
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
            %interp_factor是跟踪器的自适应速率
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;%model_xf为上一帧的，xf为这一帧的
		end

		%save position and timing保存每帧中的目标位置和处理所耗时间
		positions(frame,:) = pos;
		time = time + toc();

		%visualization可视化
		if show_visualization,
			box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];%什么意思
			stop = update_visualization(frame, box);
			if stop, break, end  %user pressed Esc, stop early
			
			drawnow
% 			pause(0.05)  %uncomment to run slower
		end
		
	end

	if resize_image,
		positions = positions * 2;
	end
end

