/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/


#include <dest/core/config.h>
#include <mat.h>
#ifdef DEST_WITH_OPENCV

#include <dest/io/database_io.h>
#include <dest/util/log.h>
#include <dest/util/draw.h>
#include <dest/util/convert.h>
#include <dest/util/glob.h>
#include <dest/io/rect_io.h>
#include <opencv2/opencv.hpp>
#include <fstream>

namespace dest {
    namespace io {
        
        ImportParameters::ImportParameters() {
            maxImageSideLength = std::numeric_limits<int>::max();
            generateVerticallyMirrored = false;
        }

        DatabaseType importDatabase(const std::string & directory,
                                    const std::string &rectangleFile,
                                    std::vector<core::Image>& images,
                                    std::vector<core::Shape>& shapes,
                                    std::vector<core::Rect>& rects,
                                    const ImportParameters & opts,
                                    std::vector<float> *scaleFactors)
        {
			const bool isAFLW = util::findFilesInDir(directory, "mat", true, true).size() > 0;

            if (isAFLW) {
				bool ok = importAFLWAnnotatedFaceDatabase(directory, rectangleFile, images, shapes, rects, opts, scaleFactors);
				return ok ? DATABASE_AFLW : DATABASE_ERROR;
			}
			else{
				DEST_LOG("Unknown database format.");
                return DATABASE_ERROR;
            }
        }
        
        bool imageNeedsScaling(cv::Size s, const ImportParameters &p, float &factor) {
            int maxLen = std::max<int>(s.width, s.height);
            if (maxLen > p.maxImageSideLength) {
                factor = static_cast<float>(p.maxImageSideLength) / static_cast<float>(maxLen);
                return true;
            } else {
                factor = 1.f;
                return false;
            }
        }
        
        void scaleImageShapeAndRect(cv::Mat &img, core::Shape &s, core::Rect &r, float factor) {
            cv::resize(img, img, cv::Size(0,0), factor, factor, CV_INTER_CUBIC);
            s *= factor;
            r *= factor;
        }

        cv::Mat loadImageFromFilePrefix(const std::string &prefix) {
            const std::string extensions[] = { ".png", ".jpg", ".jpeg", ".bmp", ""};

            cv::Mat img;
            const std::string *ext = extensions;

            do {
                img = cv::imread(prefix + *ext, cv::IMREAD_GRAYSCALE);
                ++ext;
            } while (*ext != "" && img.empty());

            return img;
        }
        
        void mirrorImageShapeAndRectVertically(cv::Mat &img,
                                               core::Shape &s,
                                               core::Rect &r,
                                               const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &permLandmarks,
                                               const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> &permRectangle)
        {
            cv::flip(img, img, 1);
            for (core::Shape::Index i = 0; i < s.cols(); ++i) {
                s(0, i) = static_cast<float>(img.cols - 1) - s(0, i);
            }
            s = (s * permLandmarks).eval();
            
            
            for (core::Rect::Index i = 0; i < r.cols(); ++i) {
                r(0, i) = static_cast<float>(img.cols - 1) - r(0, i);
            }
            
            r = (r * permRectangle).eval();
        }
        
        Eigen::PermutationMatrix<Eigen::Dynamic> createPermutationMatrixForMirroredRectangle() {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(4);
            perm.setIdentity();
            Eigen::PermutationMatrix<Eigen::Dynamic>::IndicesType &ids = perm.indices();
            
            std::swap(ids(0), ids(1));
            std::swap(ids(2), ids(3));
            
            return perm;
        }
        
        const Eigen::PermutationMatrix<Eigen::Dynamic> &permutationMatrixForMirroredRectangle() {
            const static Eigen::PermutationMatrix<Eigen::Dynamic> _instance = createPermutationMatrixForMirroredRectangle();
            return _instance;
        }

        
        Eigen::PermutationMatrix<Eigen::Dynamic> createPermutationMatrixForMirroredIBug() {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(68);
            perm.setIdentity();
            //return perm;
            Eigen::PermutationMatrix<Eigen::Dynamic>::IndicesType &ids = perm.indices();
            
            // http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
            
            // Contour
            std::swap(ids(0), ids(16));
            std::swap(ids(1), ids(15));
            std::swap(ids(2), ids(14));
            std::swap(ids(3), ids(13));
            std::swap(ids(4), ids(12));
            std::swap(ids(5), ids(11));
            std::swap(ids(6), ids(10));
            std::swap(ids(7), ids(9));
            std::swap(ids(8), ids(8));
            
            // Eyebrow
            std::swap(ids(17), ids(26));
            std::swap(ids(18), ids(25));
            std::swap(ids(19), ids(24));
            std::swap(ids(20), ids(23));
            std::swap(ids(21), ids(22));
            
            // Nose
            std::swap(ids(27), ids(27));
            std::swap(ids(28), ids(28));
            std::swap(ids(29), ids(29));
            std::swap(ids(30), ids(30));
            
            std::swap(ids(31), ids(35));
            std::swap(ids(32), ids(34));
            std::swap(ids(33), ids(33));
            
            // Eye
            std::swap(ids(39), ids(42));
            std::swap(ids(38), ids(43));
            std::swap(ids(37), ids(44));
            std::swap(ids(36), ids(45));
            std::swap(ids(40), ids(47));
            std::swap(ids(41), ids(46));
            
            // Mouth
            std::swap(ids(48), ids(54));
            std::swap(ids(49), ids(53));
            std::swap(ids(50), ids(52));
            std::swap(ids(51), ids(51));
            
            std::swap(ids(59), ids(55));
            std::swap(ids(58), ids(56));
            std::swap(ids(57), ids(57));
            
            std::swap(ids(60), ids(64));
            std::swap(ids(61), ids(63));
            std::swap(ids(62), ids(62));
            
            std::swap(ids(67), ids(65));
            std::swap(ids(66), ids(66));
            
            return perm;
        }
        
        const Eigen::PermutationMatrix<Eigen::Dynamic> &permutationMatrixForMirroredIBug() {
            const static Eigen::PermutationMatrix<Eigen::Dynamic> _instance = createPermutationMatrixForMirroredIBug();
            return _instance;
        }
        
		bool parseMatFile(const std::string& fileName, core::Shape &s) {
			//need to upgrade...
			
			MATFile *pmatFile = matOpen(fileName.data(), "r");
			if (pmatFile == NULL) {
				DEST_LOG("MatOpen error!!!");
				return false;
			}
			mxArray* pt3d_68 = matGetVariable(pmatFile, "pt3d_68");
			int numPoints = static_cast<int>(mxGetN(pt3d_68));
			REAL32_T* data = (REAL32_T*)mxGetPr(pt3d_68);
			s.resize(3, numPoints);
			s.fill(0);
			for (int i = 0; i < numPoints; ++i) {
				if (!*(data + i * 3) | !*(data + i * 3 + 1) | !*(data + i * 3 + 2)) {
					DEST_LOG("Failed to read points.");
					return false;
				}
				float temp1 = *(data + i * 3);
				float temp2 = *(data + i * 3 + 1);
				float temp3 = *(data + i * 3 + 2);
				s(0, i) = temp1; // Matlab to C++ offset
				s(1, i) = temp2;
				s(2, i) = temp3;
			}

			matClose(pmatFile);
			return true;
		}


		
		bool importAFLWAnnotatedFaceDatabase(const std::string &directory,
			const std::string &rectangleFile,
			std::vector<core::Image> &images,
			std::vector<core::Shape> &shapes,
			std::vector<core::Rect> &rects,
			const ImportParameters &opts,
			std::vector<float> *scaleFactors)
		{

			std::vector<std::string> paths = util::findFilesInDir(directory, "mat", true, true);
			DEST_LOG("Loading AFLW database. Found " << paths.size() << " candidate entries.");
			//need to upgrade...
			std::vector<core::Rect> loadedRects;
			io::importRectangles(rectangleFile, loadedRects);

			if (loadedRects.empty()) {
				DEST_LOG("No rectangles found, using tight axis aligned bounds.");
			}
			else {
				if (paths.size() != loadedRects.size()) {
					DEST_LOG("Mismatch between number of shapes in database and rectangles found.");
					return false;
				}
			}

			size_t initialSize = images.size();

			for (size_t i = 0; i < paths.size(); ++i) {
				const std::string fileNameMat = paths[i] + ".mat";

				core::Shape s;
				core::Rect r;
				bool ptsOk = parseMatFile(fileNameMat, s);
				cv::Mat cvImg = loadImageFromFilePrefix(paths[i]);
				const bool validRect = loadedRects.empty() || !loadedRects[i].isZero();

				if (ptsOk && !cvImg.empty() && validRect) {

					if (loadedRects.empty()) {
						r = core::shapeBounds(s);
					}
					else {
						r = loadedRects[i];
					}

					float f;
					if (imageNeedsScaling(cvImg.size(), opts, f)) {
						scaleImageShapeAndRect(cvImg, s, r, f);
					}

					core::Image img;
					util::toDest(cvImg, img);

					images.push_back(img);
					shapes.push_back(s);
					rects.push_back(r);

					if (scaleFactors) {
						scaleFactors->push_back(f);
					}


					if (opts.generateVerticallyMirrored) {
						cv::Mat cvFlipped = cvImg.clone();
						mirrorImageShapeAndRectVertically(cvFlipped, s, r, permutationMatrixForMirroredIBug(), permutationMatrixForMirroredRectangle());

						core::Image imgFlipped;
						util::toDest(cvFlipped, imgFlipped);

						images.push_back(imgFlipped);
						shapes.push_back(s);
						rects.push_back(r);

						if (scaleFactors) {
							scaleFactors->push_back(f);
						}
					}
				}
			}

			DEST_LOG("Successfully loaded " << (shapes.size() - initialSize) << " entries from database.");
			return (shapes.size() - initialSize) > 0;

		}

    }
}

#endif