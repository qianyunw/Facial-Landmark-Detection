/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/dest.h>

#include <dest/face/face_detector.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <random>
#include <opencv2/opencv.hpp>
#include <tclap/CmdLine.h>
#include "mat.h"

/**
    Sample program to predict shape landmarks on images.

    This program takes an image, a learnt tracker and an OpenCV face detector to
    compute shape landmark positions. OpenCV face detection works based on Viola
    Jones algorithm which requires a training phase as well. Suitable classifier 
    files can be donwloaded from OpenCV or from the dest/etc directory.

    Use any key to cycle throgh tracker cascades and see incremetal updates. Start
    configuration of shape landmarks is given in red.

    Note that his example uses an OpenCV face detector based on Viola Jones to
    provide the initial face rectangle. Therefore, you should only trackers that
    have been trained on the same input.

*/
int main(int argc, char **argv)
{
    struct {
       // std::string detector;
        std::string tracker;
        std::string image;
    } opts;

    try {
        TCLAP::CmdLine cmd("Test regressor on a single image.", ' ', "0.9");
        TCLAP::ValueArg<std::string> trackerArg("t", "tracker", "Trained tracler to load", true, "dest.bin", "file");
    //    TCLAP::ValueArg<std::string> detectorArg("d", "detector", "OpenCV face detector to load", true, "cascade.xml", "string");        
        TCLAP::UnlabeledValueArg<std::string> imageArg("image", "Image to align", true, "img.png", "file");

      //  cmd.add(&detectorArg);
        cmd.add(&trackerArg);
        cmd.add(&imageArg);

        cmd.parse(argc, argv);

      //  opts.detector = detectorArg.getValue();
        opts.tracker = trackerArg.getValue();
        opts.image = imageArg.getValue();
    }
    catch (TCLAP::ArgException &e) {
        std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
        return -1;
    }

    cv::Mat imgCV = cv::imread(opts.image, cv::IMREAD_GRAYSCALE);
    if (imgCV.empty()) {
        std::cout << "Failed to load image." << std::endl;
        return 0;
    }

    dest::core::Image img;
    dest::util::toDest(imgCV, img);

  //  dest::face::FaceDetector fd;
  //  if (!fd.loadClassifiers(opts.detector)) {
  //      std::cout << "Failed to load classifiers." << std::endl;
  //      return 0;
  //  }
    
    dest::core::Tracker t;
    if (!t.load(opts.tracker)) {
        std::cout << "Failed to load tracker." << std::endl;
        return 0;
    }

    dest::core::Rect r;

	/*
    if (!fd.detectSingleFace(img, r)) {
        std::cout << "Failed to detect face" << std::endl;
		r = dest::core::createRectangle(Eigen::Vector2f(183.0f, 108.0f), Eigen::Vector2f(399.0f, 324.0f));
		//183.0f, 399.0f, 108.0f, 324.0f
		//return 0;
    }
	*/
	//******************************************

	r = dest::core::createRectangle(Eigen::Vector2f(110.0f, 110.0f), Eigen::Vector2f(340.0f, 340.0f));
	std::cout << r << std::endl;
	std::cout << "hi" << std::endl;
	

	
	//******************************************
    // Default inverse shape normalization. Needs to be equivalent to training.
	dest::core::Shape unit_rect_to_shape;
	unit_rect_to_shape.resize(3, 4);
	dest::core::Rect unit = dest::core::unitRectangle();
	unit_rect_to_shape(0, 0) = unit(0, 0);
	unit_rect_to_shape(0, 1) = unit(0, 1);
	unit_rect_to_shape(0, 2) = unit(0, 2);
	unit_rect_to_shape(0, 3) = unit(0, 3);
	unit_rect_to_shape(1, 0) = unit(1, 0);
	unit_rect_to_shape(1, 1) = unit(1, 1);
	unit_rect_to_shape(1, 2) = unit(1, 2);
	unit_rect_to_shape(1, 3) = unit(1, 3);

	dest::core::Shape r_to_Shape;
	r_to_Shape.resize(3, 4);
	r_to_Shape(0, 0) = r(0, 0);
	r_to_Shape(0, 1) = r(0, 1);
	r_to_Shape(0, 2) = r(0, 2);
	r_to_Shape(0, 3) = r(0, 3);
	r_to_Shape(1, 0) = r(1, 0);
	r_to_Shape(1, 1) = r(1, 1);
	r_to_Shape(1, 2) = r(1, 2);
	r_to_Shape(1, 3) = r(1, 3);

	dest::core::ShapeTransform shapeToImage = dest::core::estimateSimilarityTransform(unit_rect_to_shape, r_to_Shape);

    std::vector<dest::core::Shape> steps;
    dest::core::Shape s = t.predict(img, shapeToImage, &steps);


	/*
	MATFile *pmatFile = NULL;
	mxArray *pWriteArray = NULL;

	double *outA = new double[3*68];
	for (int i = 0; i < 68; ++i) {
		*(outA + i * 3) = s(0, i);
		*(outA + i * 3 + 1) = s(1, i);
		*(outA + i * 3 + 2) = s(2, i);
	}
	pmatFile = matOpen("S.mat", "w");
	pWriteArray = mxCreateDoubleMatrix(3, 68, mxREAL);
	mxSetData(pWriteArray, outA);
	matPutVariable(pmatFile, "S", pWriteArray);
	matClose(pmatFile);
	*/






	MATFile *pmat;
	mxArray *pa1;
	double data[204];
	for (int i = 0; i < 68; ++i) {
		data[i * 3] = s(0, i);
		data[i * 3 + 1] = s(1, i);
		data[i * 3 + 2] = s(2, i);
	}
	//data = { 1.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0 };
	const char *file = "S.mat";
	int status;

	printf("Creating file %s...\n\n", file);
	pmat = matOpen(file, "w");
	if (pmat == NULL) {
		printf("Error creating file %s\n", file);
		printf("(Do you have write permission in this directory?)\n");
		return(EXIT_FAILURE);
	}

	pa1 = mxCreateDoubleMatrix(3, 68, mxREAL);
	if (pa1 == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	status = matPutVariable(pmat, "LocalDouble", pa1);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	/*
	* Ooops! we need to copy data before writing the array.  (Well,
	* ok, this was really intentional.) This demonstrates that
	* matPutVariable will overwrite an existing array in a MAT-file.
	*/
	memcpy((void *)(mxGetPr(pa1)), (void *)data, sizeof(data));
	status = matPutVariable(pmat, "LocalDouble", pa1);
	if (status != 0) {
		printf("%s :  Error using matPutVariable on line %d\n", __FILE__, __LINE__);
		return(EXIT_FAILURE);
	}

	/* clean up */
	mxDestroyArray(pa1);

	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(EXIT_FAILURE);
	}

	/*
	* Re-open file and verify its contents with matGetVariable
	*/
	pmat = matOpen(file, "r");
	if (pmat == NULL) {
		printf("Error reopening file %s\n", file);
		return(EXIT_FAILURE);
	}

	/* clean up before exit */
	mxDestroyArray(pa1);

	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n", file);
		return(EXIT_FAILURE);
	}
	printf("Done\n");















    bool done = false;
    size_t id = 0;
    while (!done) {
        
        cv::Scalar color = (id == steps.size() - 1) ? cv::Scalar(255, 0, 102) : cv::Scalar(255, 255, 255);
        cv::Mat tmp = dest::util::drawShape(img, steps[id], color);
        cv::imshow("prediction", tmp);

        id = (id + 1) % steps.size();

        int key = cv::waitKey();
        if (key == 'x')
            done = true;
    }


    
    return 0;
}
