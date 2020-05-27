function kcf
% ncc VOT integration example
% 
% This function is an example of tracker integration into the toolkit.
% The implemented tracker is a very simple NCC tracker that is also used as
% the baseline tracker for challenge entries.
%

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('rectangle');

% Initialize the tracker and params
params.kernel_type = 'gaussian';
params.feature_type = 'hog';
features.gray = false;
features.hog = false;

params.padding = 1.5;  %extra area surrounding the target

params.lambda = 1e-4;  %regularization
params.output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)

params.interp_factor = 0.02;

kernel.sigma = 0.5;

kernel.poly_a = 1;
kernel.poly_b = 9;

features.hog = true;
features.hog_orientations = 9;
params.cell_size = 4;



[state, ~, params] = kcf_initialize(imread(image), region, params, features, kernel);

while true

    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);

    if isempty(image)
        break;
    end;

	% Perform a tracking step, obtain new region
    [state, region, params] = kcf_update(state, imread(image), params, features, kernel);

    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region);

end;

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

function [state, location, params] = kcf_initialize(I, region, params, features, kernel)

    % =====VOT part=====
    gray = double(rgb2gray(I));

    [height, width] = size(gray);

    % If the provided region is a polygon ...
    if numel(region) > 4
        x1 = round(min(region(1:2:end)));
        x2 = round(max(region(1:2:end)));
        y1 = round(min(region(2:2:end)));
        y2 = round(max(region(2:2:end)));
        region = round([x1, y1, x2 - x1, y2 - y1]);
    else
        region = round([round(region(1)), round(region(2)), ... 
            round(region(1) + region(3)) - round(region(1)), ...
            round(region(2) + region(4)) - round(region(2))]);
    end;

    x1 = max(1, region(1));
    y1 = max(1, region(2));
    x2 = min(width-2, region(1) + region(3) - 1);
    y2 = min(height-2, region(2) + region(4) - 1);

    template = gray((y1:y2)+1, (x1:x2)+1);

%     state = struct('template', template, 'size', [x2 - x1 + 1, y2 - y1 + 1]);
    state = struct('template', template, 'size', [y2 - y1 + 1, x2 - x1 + 1]);
    state.window = max(state.size) * 2;
%     state.position = [x1 + x2 + 1, y1 + y2 + 1] / 2;
    state.position = [y1 + y2 + 1, x1 + x2 + 1] / 2;

    location = [x1, y1, state.size];

    % =====KCF part=====
    target_sz = state.size;
    params.pos = state.position;
    
    params.resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if params.resize_image
		params.pos = floor(params.pos / 2);
		target_sz = floor(target_sz / 2);
    end
    
    params.window_sz = floor(target_sz * (1 + params.padding)); 
    
    %create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
    output_sigma = sqrt(prod(target_sz)) * params.output_sigma_factor / params.cell_size;
	params.yf = fft2(gaussian_shaped_labels(output_sigma, floor(params.window_sz / params.cell_size)));
    params.cos_window = hann(size(params.yf,1)) * hann(size(params.yf,2))';

    
    if params.resize_image
        params = kcf_modelUpdate(imresize(gray,0.5), params, 1, features, kernel);
    else
        params = kcf_modelUpdate(gray, params, 1, features, kernel);
    end

end

function [state, location, params] = kcf_update(state, I, params, features, kernel)
   
    % =====VOT part=====
    gray = double(rgb2gray(I)) ; 

    
    % =====KCF part=====
  
    if params.resize_image
        gray = imresize(gray, 0.5);
    end
 
    patch = get_subwindow(gray, params.pos, params.window_sz);
	zf = fft2(get_features(patch, features, params.cell_size, params.cos_window));
   
    switch params.kernel_type
        case 'gaussian',
            kzf = gaussian_correlation(zf, params.model_xf, kernel.sigma);
        case 'polynomial',
            kzf = polynomial_correlation(zf, params.model_xf, kernel.poly_a, kernel.poly_b);
        case 'linear',
            kzf = linear_correlation(zf, params.model_xf);
    end
    response = real(ifft2(params.model_alphaf .* kzf));  %equation for fast detection

    % target location
    [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
    if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
        vert_delta = vert_delta - size(zf,1);
    end
    if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
        horiz_delta = horiz_delta - size(zf,2);
    end

    params.pos = params.pos + params.cell_size * [vert_delta - 1, horiz_delta - 1];

    % model update
    params = kcf_modelUpdate(gray, params, 0, features, kernel);
   
    % =====VOT part=====
    if params.resize_image
        state.position = params.pos*2;
    else
        state.position = params.pos;
    end
    location = [state.position - state.size/2 , state.size];
    location = location([2,1,4,3]);
    state.position = state.position([2,1]);

end

function [params] = kcf_modelUpdate(im, params, firstframe, features, kernel)

    %obtain a subwindow for training at newly estimated target position
    patch = get_subwindow(im, params.pos, params.window_sz);
    xf = fft2(get_features(patch, features, params.cell_size, params.cos_window));

    %Kernel Ridge Regression, calculate alphas (in Fourier domain)
    switch params.kernel_type
    case 'gaussian',
        kf = gaussian_correlation(xf, xf, kernel.sigma);
    case 'polynomial',
        kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
    case 'linear',
        kf = linear_correlation(xf, xf);
    end
    alphaf = params.yf ./ (kf + params.lambda);   %equation for fast training

    if firstframe == 1
        params.model_alphaf = alphaf;
        params.model_xf = xf;
    else
        %subsequent frames, interpolate model
        params.model_alphaf = (1 - params.interp_factor) * params.model_alphaf + params.interp_factor * alphaf;
        params.model_xf = (1 - params.interp_factor) * params.model_xf + params.interp_factor * xf;
    end

end