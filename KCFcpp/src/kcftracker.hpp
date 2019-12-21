/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: horizontal area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#pragma once

#include "tracker.h"

#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif

class KCFTracker : public Tracker
{
public:
    // Constructor  构造KCF跟踪器的类
    // 使用hog特征 使用固定窗口大小 使用多尺度 使用lab色空间特征
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    // Initialize tracker  初始化跟踪器， roi 是目标初始框的引用， image 是进入跟踪的第一帧图像
    virtual void init(const cv::Rect &roi, cv::Mat image);
    
    // Update position based on the new frame  使用新一帧更新图像， image 是新一帧图像
    virtual cv::Rect update(cv::Mat image);

    float interp_factor; // linear interpolation factor for adaptation  自适应的线性插值因子，会因为hog，lab的选择而变化
    float sigma; // gaussian kernel bandwidth   高斯卷积核带宽，会因为hog，lab的选择而变化
    float lambda; // regularization  正则化，0.0001
    int cell_size; // HOG cell size  HOG元胞数组尺寸，4
    int cell_sizeQ; // cell size^2, to avoid repeated operations  元胞数组内像素数目，16，为了计算省事
    float padding; // extra area surrounding the target  目标扩展出来的区域，2.5
    float output_sigma_factor; // bandwidth of gaussian target  高斯目标的带宽，不同hog，lab会不同
    int template_size; // template size  模板大小，在计算_tmpl_sz时，较大变成被归一成96，而较小边长按比例缩小
    float scale_step; // scale step for multi-scale estimation  多尺度估计的时候的尺度步长
    float scale_weight;  // to downweight detection scores of other scales for added stability
                                           // 为了增加其他尺度检测时的稳定性，给检测结果峰值做一定衰减，为原来的0.95倍

protected:
    // Detect object in the current frame.
    // 检测当前帧的目标
    //z是前一帧的训练或第一帧的初始化结果， x是当前帧当前尺度下的特征， peak_value是检测结果峰值
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);

    // train tracker with a single image
    // 使用当前图像的检测结果进行训练  x是当前帧当前尺度下的特征， train_interp_factor是interp_factor
    void train(cv::Mat x, float train_interp_factor);

    // Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y,
    //which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
    // 使用带宽SIGMA计算高斯卷积核以用于所有图像X和Y之间的相对位移
    // 必须都是MxN大小。二者必须都是周期的（即，通过一个cos窗口进行预处理）
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

    // Create Gaussian Peak. Function called only in the first frame.      创建高斯峰函数，函数只在第一帧的时候执行
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    // 从图像得到子窗口，通过赋值填充并检测特征
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

    // Initialize Hanning window. Function called only in the first frame.     初始化hanning窗口。函数只在第一帧被执行。
    void createHanningMats();

    // Calculate sub-pixel peak for one dimension      计算一维亚像素峰值
    float subPixelPeak(float left, float center, float right);

    cv::Mat _alphaf;  // 初始化/训练结果alphaf，用于检测部分中结果的计算
    cv::Mat _prob;     // 初始化结果prob，不再更改，用于训练
    cv::Mat _tmpl;    // 初始化/训练的结果，用于detect的z
    cv::Mat _num;    // 貌似都被注释掉了
    cv::Mat _den;     // 貌似都被注释掉了
    cv::Mat _labCentroids;    // lab质心数组

private:
    int size_patch[3];    // hog特征的sizeY，sizeX，numFeatures
    cv::Mat hann;     // createHanningMats()的计算结果
    cv::Size _tmpl_sz;     // hog元胞对应的数组大小
    float _scale;    // 修正成_tmpl_sz后的尺度大小
    int _gaussian_size;    // 未引用？？？
    bool _hogfeatures;    // hog标志位
    bool _labfeatures;     // lab标志位
};
