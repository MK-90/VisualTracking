function video_path = choose_video(base_path)

% video_path = choose_video(base_path)

%process path to make sure it's uniform 把base_path 中的 \ 换成 / ，如果路径最后没有 / ,添加上
if ispc(), base_path = strrep(base_path, '\', '/'); end  %newStr = strrep(str,old,new) 将 str 中出现的所有 old 都替换为 new。
if base_path(end) ~= '/', base_path(end+1) = '/'; end

%list all sub-folders
contents = dir(base_path);
names = {};
for k = 1:numel(contents),
    name = contents(k).name;
    if isdir([base_path name]) && ~strcmp(name, '.') && ~strcmp(name, '..'),
        names{end+1} = name;  %#ok
    end
end

%no sub-folders found
if isempty(names), video_path = []; return; end

%choice GUI
choice = listdlg('ListString',names, 'Name','Choose video', 'SelectionMode','single');

if isempty(choice),  %user cancelled
    video_path = [];
else
    video_path = [base_path names{choice} '/'];
end

end

