% example evaluation code for color naming on ebay data set

image_path='/home/joost/images/ebay/';       % set path to ebay data set location

load('w2c.mat');
[max1,w2CNI]=max(w2c,[],2);      % w2cCNI contains the color-name-index ranging from 1-11 

cd(image_path);
fid=fopen(sprintf('test_images.txt'),'r');
fid2=fopen(sprintf('mask_images.txt'),'r');

image_counter=0;

while(~feof(fid))   % read images and masks
    image_counter=image_counter+1;
    image_name=sprintf('%s',fgetl(fid));
    mask_name=sprintf('%s',fgetl(fid2));
    im=double(imread(image_name));
    mask=(imread(mask_name)>0);
          
    % compute color name indexes for all pixels in mask
    RR=im(:,:,1);GG=im(:,:,2);BB=im(:,:,3);
    index_im = 1+floor(RR(mask)/8)+32*floor(GG(mask)/8)+32*32*floor(BB(mask)/8);
    CNI=w2CNI(index_im(:));
    
    % compute percentage correctly classified pixels
    image_label=ceil((mod( (image_counter-1) ,110)+1)/10);
    class_score(image_counter) = sum(CNI==image_label)/length(CNI);
end
fclose(fid);
fclose(fid2);

fprintf('The average percentage of correctly classified pixels is %f\n', mean(class_score));