function [precision, fps] = run_tracker(video, kernel_type, feature_type, show_visualization, show_plots)
    %�ݹ����
	%path to the videos (you'll be able to choose one with the GUI).
	base_path = 'F:\Postgraduate Learning\OTB\TB50\';
    %base_path = 'G:\data\tracking\VOT\vot2016\';
	%default settings  Ĭ������
	if nargin < 1, video = 'choose'; end%�����������ĸ���С��1������ô��video��Ĭ��ֵ��ִֻ����һ�����
	if nargin < 2, kernel_type = 'linear'; end%nargin�ľ��嶨��
	if nargin < 3, feature_type = 'hogcolor'; end
	if nargin < 4, show_visualization = ~strcmp(video, 'all'); end
	if nargin < 5, show_plots = ~strcmp(video, 'all'); end

	%parameters according to the paper. at this point we can override parameters based on the chosen kernel or feature type
    %���������ļ�������һ���ϣ����ǿ��Ը���ѡ����ں˻������͸��ǲ���
	
	kernel.type = kernel_type;
	
	features.gray = false;
	features.hog = false;
    features.hogcolor = false;
	
	padding = 1.5;  %���ٿ����䱶��
	lambda = 1e-4;  %regularization ���滯 ģ�͸��²���
	output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
                                %�ռ������Ŀ������ȣ�
	switch feature_type
	case 'gray',
		interp_factor = 0.075;  %linear interpolation factor for adaptation
                                %������Ӧ�������ڲ�����
		kernel.sigma = 0.2;     %gaussian kernel bandwidth
                                %��˹�ں˴���		
		kernel.poly_a = 1;      %polynomial kernel additive term
                                %����ʽ�˼ӷ���
		kernel.poly_b = 7;      %polynomial kernel exponent
                                %����ʽ�ں�ָ��
		features.gray = true;
		cell_size = 1;
		
	case 'hog',
		interp_factor = 0.02;  %interp_factor�Ǹ�����������Ӧ���ʣ�������ģ�͵ļ�������
		
		kernel.sigma = 0.5; %��˹�ں˴���		
		
		kernel.poly_a = 1; %����ʽ�˼ӷ���
		kernel.poly_b = 9; %����ʽ�ں�ָ��
		
		features.hog = true;
		features.hog_orientations = 9;
		cell_size = 4;
    case 'hogcolor',
		interp_factor = 0.01;  %interp_factor�Ǹ�����������Ӧ���ʣ�������ģ�͵ļ�������
		
		kernel.sigma = 0.5; %��˹�ں˴���		
		
		kernel.poly_a = 1; %����ʽ�˼ӷ���
		kernel.poly_b = 9; %����ʽ�ں�ָ��
		
		features.hogcolor = true;
		features.hog_orientations = 9;
		cell_size = 4;
		
	otherwise
		error('Unknown feature.')
	end

	assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')
    %��matlab��assert���������ж�һ��expression�Ƿ����
    %strcmp���������ַ����Ƚϵĺ���

	switch video
	case 'choose',
		%ask the user for the video, then call self with that video name.
        %���û�ѯ����Ƶ��Ȼ��ʹ�ø���Ƶ���ƺ����Լ���
		video = choose_video(base_path);
		if ~isempty(video),%isempty(msg)�ж�msg�Ƿ�Ϊ�գ����Ϊ�գ����Ϊ1������Ϊ0.
			[precision, fps] = run_tracker(video, kernel_type, ...
				feature_type, show_visualization, show_plots);%���ñ����ݹ���á�
			%[precision, fps] = run_tracker(video, kernel_type, feature_type, show_visualization, show_plots)
			if nargout == 0,  %don't output precision as an argument
				clear precision
			end
		end
		
		
	case 'all',
		%all videos, call self with each video name.
		
		%only keep valid directory names ֻ������Ч��Ŀ¼��
		dirs = dir(base_path);
		videos = {dirs.name};
		videos(strcmp('.', videos) | strcmp('..', videos) | ...
			strcmp('anno', videos) | ~[dirs.isdir]) = [];
		
		%the 'Jogging' sequence has 2 targets, create one entry for each.
        %�����ܡ�������2��Ŀ�꣬Ϊÿ��Ŀ�괴��һ����Ŀ��
		%we could make this more general if multiple targets per video becomes a common occurence.
		%���ÿ����Ƶ�Ķ��Ŀ���Ϊ��������������ǿ���ʹ���Ϊͨ�á�
		videos(strcmpi('Jogging', videos)) = [];
		videos(end+1:end+2) = {'Jogging.1', 'Jogging.2'};
		
		all_precisions = zeros(numel(videos),1);  %to compute averages
		all_fps = zeros(numel(videos),1);
		
		if ~exist('matlabpool', 'file'),
			%no parallel toolbox, use a simple 'for' to iterate
            %û��ƽ�еĹ����䣬ʹ��һ���򵥵�'for'������
			for k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, ...
					kernel_type, feature_type, show_visualization, show_plots);
			end
		else
			%evaluate trackers for all videos in parallel
            %��������������Ƶ�ĸ�����
			if parpool('size') == 0,
				parpool open;
			end
			parfor k = 1:numel(videos),
				[all_precisions(k), all_fps(k)] = run_tracker(videos{k}, ...
					kernel_type, feature_type, show_visualization, show_plots);
			end
		end
		
		%compute average precision at 20px, and FPS
        %����20px��ƽ�����Ⱥ�FPS     px������
		mean_precision = mean(all_precisions);
		fps = mean(all_fps);
		fprintf('\nAverage precision (20px):% 1.3f, Average FPS:% 4.2f\n\n', mean_precision, fps)
		if nargout > 0
			precision = mean_precision;
		end
		
		
	case 'benchmark'
		%running in benchmark mode - this is meant to interface easily with the benchmark's code.
		%�Ի�׼ģʽ���� - ����ζ�ſ������ɵ����׼����������ӡ�
		
	%get information (image file names, initial position, etc) fromthe benchmark's workspace variables
	%�ӻ�׼�Ĺ����ռ������ȡ��Ϣ��ͼ���ļ�������ʼλ�õȣ�
		seq = evalin('base', 'subS');%base��MATLAB�Ļ��������ռ䡣��subS��ֵ����seq��
		target_sz = seq.init_rect(1,[4,3]);
		pos = seq.init_rect(1,[2,1]) + floor(target_sz/2);

        %��ʼ��λĿ���λ��
		img_files = seq.s_frames;
		video_path = [];
		
		%call tracker function with all the relevant parameters
        %����������ز����ĺ��и��ٺ���
		positions = tracker(video_path, img_files, pos, target_sz, ...
			padding, kernel, lambda, output_sigma_factor, interp_factor, ...
			cell_size, features, false);
		%return results to benchmark, in a workspace variable
        %��������ص���׼���ڹ����ռ������
		rects = [positions(:,2) - target_sz(2)/2, positions(:,1) - target_sz(1)/2];
		rects(:,3) = target_sz(2);
		rects(:,4) = target_sz(1);
		res.type = 'rect';
		res.res = rects;
		assignin('base', 'res', res);
		
		
	 otherwise
		%we were given the name of a single video to process.
        %���ǻ����һ��Ҫ�������Ƶ�����ơ�
		%get image file names, initial state, and ground truth for evaluation
       %��ȡͼ���ļ�������ʼ״̬�͵���ʵ����������
        
		[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(base_path, video);
		%[ground_truth,img_files, pos, target_sz, video_path] = load_video_info_votformat(base_path, video);
		
		%call tracker function with all the relevant parameters
		[positions, time] = tracker(video_path, img_files, pos, target_sz, ...
			padding, kernel, lambda, output_sigma_factor, interp_factor, ...
			cell_size, features, show_visualization);
		
		
		%calculate and show precision plot, as well as frames-per-second
        %�������ʾ��ȷ��ͼ���Լ�ÿ��֡��
		precisions = precision_plot(positions, ground_truth, video, show_plots);
        %disp(precisions);
		fps = numel(img_files) / time;

		fprintf('%12s - Precision (20px):% 1.3f, FPS:% 4.2f\n', video, precisions(20), fps)
      
		if nargout > 0,
			%return precisions at a 20 pixels threshold ����20������ֵ�ľ���
			precision = precisions(20);
		end

	end
end
