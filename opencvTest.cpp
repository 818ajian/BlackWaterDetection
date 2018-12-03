#include <iostream>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/opengl.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace cv;
using namespace cv::cuda;
const float THRESHOLD = 120;


int main() {
    // Replace the file name with 0 for captureing the live camera feed
    VideoCapture cap("emission.mkv"); 

    if (!cap.isOpened())
        return -1;
    // Declaring GPU Mat for holding the frame
    GpuMat g;
    Mat frame;
    Mat original_frame;
    float image_threshold;

    bool grabFrame = true;
    while (grabFrame) {
        // Grab frame
        cap >> frame;
        cap >> original_frame;

        if (frame.empty() || original_frame.empty())
            {
                cout<<"Video Ended Successfully"<<endl;
                break;
            }

        frame.convertTo(frame, CV_32F);
        // Upload to gpu
        g.upload(frame);

        // convert to normalized float
        g.convertTo(g, CV_32F, 1.f / 255);
        cv::cuda::GpuMat image(g);

        //Croping the image containing region of interest [top_left_X,top_left_Y, Width, Height]
        cv::Rect myROI(150,210, 200, 165);
        
        cv::cuda::GpuMat croppedImage = image(myROI);
        Mat cropped_frame;
        croppedImage.download(cropped_frame);
        
        vector<Mat> channels(3);
        split(cropped_frame, channels);
        cv::Rect rect(150,210, 200, 165);

        image_threshold = ((mean(channels[0])[0]*255)+(mean(channels[1])[0]*255)+(mean(channels[2])[0]*255))/3;

        if( image_threshold > THRESHOLD){
            cv::rectangle(original_frame,rect, cv::Scalar(0,255,0),4);
        }else{
            cv::rectangle(original_frame,rect, cv::Scalar(0,0,255),4);
        }
        
        imshow("Black Water Detection", original_frame);

        if (cv::waitKey(30) >= 0)
            grabFrame = false;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}