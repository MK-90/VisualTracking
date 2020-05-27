
% load the word to color names matrix. The words are a 32x32x32 grid on the sRGB space. 

load('w2c.mat');
tic
% first example
% im=double(imread('car.jpg'));       % load test image
im=double(imread('00000009.jpg'));       % load test image


% compute the color name assignment for all pixels in image im:
out=im2c(im,w2c,-1);                % using im2c(im,w2c,0) is much faster
toc
figure(1);
subplot(1,2,1);imshow(uint8(im));
subplot(1,2,2);imshow(uint8(out));

% second example:
% im2=double(imread('opp_color_circle.tif'));     
% 
% out2=im2c(im2,w2c,-1);   
% 
% figure(2);
% subplot(1,2,1);imshow(uint8(im2));
% subplot(1,2,2);imshow(uint8(out2));