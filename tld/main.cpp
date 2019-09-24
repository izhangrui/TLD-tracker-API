#include <opencv2/opencv.hpp>
#include <iostream>
#include "TLD.h"
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
	Rect roi;
	Mat frame, frame_gray;
	//ÊµÀý»¯¸ú×ÙÆ÷
	TLD tracker;
	VideoCapture cap(0);

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 340);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);


	cap >> frame; 
	cvtColor(frame, frame_gray, CV_RGB2GRAY);
	roi = selectROI("tracker", frame_gray);  
	if (roi.width == 0 || roi.height == 0)
		return 0; 
	//³õÊ¼»¯¸ú×ÙÆ÷
	tracker.init(frame_gray, roi);
	printf("Start the tracking process\n");
	for (;;) {
		cap >> frame;
		cvtColor(frame, frame_gray, CV_RGB2GRAY);
		if (frame.rows == 0 || frame.cols == 0)
			break; 
		//¸ú×ÙÃ¿Ò»Ö¡
		tracker.processFrame(frame_gray, roi);
		rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);
		imshow("tracker", frame);  
		if (waitKey(30) == 27)
			break;
	}
	return 0;
}