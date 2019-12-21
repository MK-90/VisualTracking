#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <unistd.h> // 自己添加的，usleep 函数用到

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){

	if (argc > 5) return -1;

    bool HOG = true;    // 是否使用hog特征
    bool FIXEDWINDOW = false;  // 是否使用修正窗口
    bool MULTISCALE = true;  // 是否使用多尺度
    bool SILENT = true;   // 是否不做显示
    bool LAB = false;  // 是否使用LAB颜色

    //不同位置的参数代表的含义
	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

    // Frame readed 当前帧
	Mat frame;

    // Tracker results  跟踪目标的结果框
	Rect result;

    // Path to list.txt  images.txt的路径，用于读取图像
	ifstream listFile;
    string fileName = "images.txt";  //图像序列列表
  	listFile.open(fileName);

  	// Read groundtruth for the 1st frame
  	ifstream groundtruthFile;
    string groundtruth = "region.txt";  //存放了第一帧中目标的位置
  	groundtruthFile.open(groundtruth);
  	string firstLine;
  	getline(groundtruthFile, firstLine);
	groundtruthFile.close();
  	
  	istringstream ss(firstLine);

    // Read groundtruth like a dumb    从给定的第一帧目标框读入四个顶点的坐标
  	float x1, y1, x2, y2, x3, y3, x4, y4;
    // region.txt 中坐标是以逗号隔开的，所以每次取一个坐标之后用ch来取出坐标之后的逗号
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4; 

    // Using min and max of X and Y for groundtruth rectangle
    //  使用四个顶点计算出目标框 , 由此来看四个点的顺序可以相互颠倒
	float xMin =  min(x1, min(x2, min(x3, x4)));
	float yMin =  min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;

	// Read Images
	ifstream listFramesFile;
	string listFrames = "images.txt";
	listFramesFile.open(listFrames);
	string frameName;

	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;

	while ( getline(listFramesFile, frameName) ){
		frameName = frameName;

		// Read each frame from the list
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		// First frame, give the groundtruth to the tracker
		if (nFrames == 0) {
			tracker.init( Rect(xMin, yMin, width, height), frame );
            rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 2, 8 );
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else{
			result = tracker.update(frame);
            rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 2, 8 );
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
            usleep(10000); //自己添加的，用来使图像更新的慢一些，便于观察
		}

		nFrames++;

		if (!SILENT){
			imshow("Image", frame);
			waitKey(1);
		}
	}
	resultsFile.close();
	listFile.close();
}
