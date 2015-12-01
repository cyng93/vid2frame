/**
 * @file objectDetection.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>
#include <iomanip>

#include <pthread.h>
#include <sys/sysinfo.h>

#define scaleInput 40        // WIDTH=16*n    HEIGHT=9*n
#define jumpFrame 30        // uncomment to detect all frame
#define outputFrame         // uncomment to disable writing frame to directory
#define taskset 16          // correct the numOfCore when program is run with taskset


using namespace std;
using namespace cv;

/** Function Headers */

void *handler(void* parameters);

RNG rng(12345);

unsigned long frameCount = 0;
unsigned long numOfTotalFrame = 0;

String videoFilename = "srcVideo.mov";

String outputFilePrefix  = "frames/negCase3_";
String outputFileType = ".png";
stringstream ss;
String outputFilename;

int numOfCores = 0;
int curThreadIndex = 0;
pthread_mutex_t threadIndexLock;
int *retVal;


/**
 * @function main
 */
int main( int argc, char **argv  )
{
    int i;
    pthread_t *threadPool;

    numOfCores = get_nprocs();
#ifdef taskset
    numOfCores = (numOfCores < taskset) ? numOfCores : taskset;
#endif

    //-- 0. Get Num Of Frames
    VideoCapture capture( videoFilename );
    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        return -1;
    }
    numOfTotalFrame = capture.get(CV_CAP_PROP_FRAME_COUNT);
    capture.release();

#ifdef jumpFrame
    unsigned long numOfSampleFrame = numOfTotalFrame/jumpFrame;
    cout << " Frame Jumping Rate : " << jumpFrame << endl;
#else
    unsigned long numOfSampleFrame = numOfTotalFrame;
#endif
    cout << " Getting Frame [ " << numOfSampleFrame << " / " << numOfTotalFrame << " ] frames ..." <<  endl;

    pthread_mutex_init(&threadIndexLock, NULL);
    threadPartialCorrect = (threadPartialCorrect_t *)malloc(numOfCores * sizeof(threadPartialCorrect_t));
    threadPool = (pthread_t *)malloc(numOfCores * sizeof(pthread_t));


    // thread creation
    for(i=0; i<numOfCores; i++)
        pthread_create(&threadPool[i], NULL, handler, NULL);
    // thread join
    for(i=0; i<numOfCores; i++)
            pthread_join(threadPool[i], NULL);


    pthread_mutex_destroy(&threadIndexLock);
    free(threadPool);
    free(threadPartialCorrect);

    return 0;
}



void * handler(void* parameters)
{
    int myThreadIndex;
    unsigned long framePerThread = numOfTotalFrame/numOfCores;
    unsigned long frameCount;

    VideoCapture capture;
    Mat frame;
    // each thread getting their own index
    pthread_mutex_lock(&threadIndexLock);
    myThreadIndex = curThreadIndex;
    curThreadIndex++;
    pthread_mutex_unlock(&threadIndexLock);

    frameCount = (myThreadIndex * framePerThread);

    //-- 1. Read the video stream
    capture.open( videoFilename );

    if( !capture.isOpened() ){
        cout << "Fail to open video file" << endl;
        *retVal = -1;
        return (void *)retVal;
    }

#ifdef scaleInput
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 16 * scaleInput);      // Ratio = 16 : 9
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 9 * scaleInput);
#endif

    for(;;){
    #ifdef jumpFrame
		if(! capture.set(CV_CAP_PROP_POS_FRAMES, frameCount += jumpFrame)) { cout << "error jumpFrame"; *retVal=-1; return (void*)retVal; }
    #else
        frameCount++;
    #endif

		if( frameCount >= (myThreadIndex+1)*framePerThread || !capture.read(frame) )  // frameCount check
            break;

    #ifdef scaleInput
		resize(frame, frame, Size( 16 * scaleInput, 9 * scaleInput), 0, 0, INTER_CUBIC);
    #endif

		if( frame.empty() ) 
		{ 
			printf(" --(!) No captured frame -- Break!"); break; 
		}
	    else
	    {
	        ss << outputFilePrefix << setfill('0') << setw(5) << frameCount << outputFileType;
	        outputFilename = ss.str();
	        ss.str("");
	        imwrite(outputFilename, frame);
	    }
    }


}
