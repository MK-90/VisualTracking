Example code on color naming as described in:
Learning Color Names for Real-World Applications
J. van de Weijer, C. Schmid, J. Verbeek, D. Larlus.
IEEE Transactions in Image Processing 2009.

Run the example_color_naming.m demo for an exmample of color naming image pixels. 
The programm calls the im2c function which annotates image pixels with color names. 
The function takes a color image as input and returns the probability for eleven color names---black, blue, 
brown, gray, green, orange, pink, purple, red, white, yellow---for all pixels in the image. 
The mapping from RGB values to color names (w2c.mat) has been learned from Google images (2500).

A second rougher quanitization of the sRGB space, w2c_4096.mat, has also been provided. 
The files are also provided in .txt format where the first three columns indicate the sRGB value and 
column 4-15 the probability over the color names.

