/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/tracker.h>
#include <dest/core/regressor.h>
#include <dest/util/log.h>
#include <dest/io/matrix_io.h>
#include <fstream>

#include <tclap/CmdLine.h>

namespace dest {
    namespace core {
        
        struct Tracker::data {
            typedef std::vector<Regressor> RegressorVector;            
            RegressorVector cascade;
            Shape meanShape;
            Shape meanShapeRectCorners;            
			
            flatbuffers::Offset<io::Tracker> save(flatbuffers::FlatBufferBuilder &fbb) const {
                flatbuffers::Offset<io::MatrixF> lmeans = io::toFbs(fbb, meanShape);
                flatbuffers::Offset<io::MatrixF> lbounds = io::toFbs(fbb, meanShapeRectCorners);

                std::vector< flatbuffers::Offset<io::Regressor> > lregs;
                for (size_t i = 0; i < cascade.size(); ++i) {
                    lregs.push_back(cascade[i].save(fbb));
                }

                auto vregs = fbb.CreateVector(lregs);

                io::TrackerBuilder b(fbb);
                b.add_cascade(vregs);
                b.add_meanShape(lmeans);
                b.add_meanShapeRectCorners(lbounds);

                return b.Finish();
            }

            void load(const io::Tracker &fbs) {

                io::fromFbs(*fbs.meanShape(), meanShape);
                io::fromFbs(*fbs.meanShapeRectCorners(), meanShapeRectCorners);

                cascade.resize(fbs.cascade()->size());
                for (flatbuffers::uoffset_t i = 0; i < fbs.cascade()->size(); ++i) {
                    cascade[i].load(*fbs.cascade()->Get(i));
                }
            }
			
        };
        
        Tracker::Tracker()
        : _data(new data())
        {
        }
        
        Tracker::Tracker(const Tracker &other)
        : _data(new data(*other._data))
        {
        }
        
        Tracker::~Tracker()
        {}
		
        flatbuffers::Offset<io::Tracker> Tracker::save(flatbuffers::FlatBufferBuilder &fbb) const
        {
            return _data->save(fbb);
        }

        void Tracker::load(const io::Tracker &fbs)
        {
            _data->load(fbs);
        }

        bool Tracker::save(const std::string &path) const
        {
            std::ofstream ofs(path, std::ofstream::binary);
            if (!ofs.is_open()) return false;

            flatbuffers::FlatBufferBuilder fbb;
            io::FinishTrackerBuffer(fbb, save(fbb));

            ofs.write(reinterpret_cast<char*>(fbb.GetBufferPointer()), fbb.GetSize());
            return !ofs.bad();
        }

        bool Tracker::load(const std::string &path)
        {
            std::ifstream ifs(path, std::ifstream::binary);
            if (!ifs.is_open()) return false;

            std::string buf;
            ifs.seekg(0, std::ios::end);
            buf.resize(static_cast<size_t>(ifs.tellg()));
            ifs.seekg(0, std::ios::beg);
            ifs.read(&buf[0], buf.size());

            if (ifs.bad())
                return false;

			//  flatbuffers::Verifier v(reinterpret_cast<const uint8_t*>(buf.data()), buf.size());
            flatbuffers::Verifier v(reinterpret_cast<const uint8_t*>(buf.data()), buf.size(), 64, 9000000000000000);
            if (!io::VerifyTrackerBuffer(v)) {
                return false;
            }

            const io::Tracker *t = io::GetTracker(buf.data());
            load(*t);

            return true;
        }
        
		//核心的训练代码竟然只有这么一段你敢信
        bool Tracker::fit(SampleData &t) {
			
            eigen_assert(!t.samples.empty());
            
			DEST_LOG("Starting to fit tracker on " << t.samples.size() << " samples.");
			DEST_LOG(t.params);

            Tracker::data &data = *_data;
            
            const int numSamples = static_cast<int>(t.samples.size());

            RegressorTraining rt;
            rt.training = &t;
            rt.numLandmarks = static_cast<int>(t.samples.front().estimate.cols());
            rt.input = t.input;
            
            
            // Re-eval mean shape here.
            rt.meanShape = Shape::Zero(3, rt.numLandmarks);
			Shape temp_1 = Shape::Zero(3, rt.numLandmarks);
			Shape temp_2 = Shape::Zero(3, rt.numLandmarks);
			Shape temp_3 = Shape::Zero(3, rt.numLandmarks);

			//加大限额
			if (numSamples < 50000) {
				for (int i = 0; i < numSamples; ++i) {
					rt.meanShape += t.samples[i].estimate;
				}
				rt.meanShape /= static_cast<float>(numSamples);
			}
			else if (numSamples < 100000) {
				for (int i = 0; i < 50000; ++i) {
					temp_1 += t.samples[i].estimate;
				}
				temp_1 /= static_cast<float>(50000);

				for (int i = 50000; i < numSamples; ++i) {
					temp_2 += t.samples[i].estimate;
				}
				temp_2 /= static_cast<float>(numSamples - 50000);
				float weight = static_cast<float>(50000) / static_cast<float>(numSamples);

				rt.meanShape = temp_1*weight + temp_2*(1 - weight);
			}
			else if (numSamples < 150000) {
				for (int i = 0; i < 50000; ++i) {
					temp_1 += t.samples[i].estimate;
				}
				temp_1 /= static_cast<float>(50000);

				for (int i = 50000; i < 100000; ++i) {
					temp_2 += t.samples[i].estimate;
				}
				temp_2 /= static_cast<float>(50000);

				for (int i = 100000; i < numSamples; ++i) {
					temp_3 += t.samples[i].estimate;
				}
				temp_3 /= static_cast<float>(numSamples - 50000);
				float weight = static_cast<float>(50000) / static_cast<float>(numSamples);

				rt.meanShape = temp_1*weight + temp_2*weight + temp_3*(1-weight-weight);

			}
			else
				DEST_LOG("Too many samples, help yourself to dest::tracker.cpp");
			/*
            for (int i = 0; i < numSamples; ++i) {
                rt.meanShape += t.samples[i].estimate;
            }
            rt.meanShape /= static_cast<float>(numSamples);
            */
            // Build cascade
            data.cascade.resize(t.params.numCascades);
            
            for (int i = 0; i < t.params.numCascades; ++i) {
				DEST_LOG("Building cascade ");
                
                // Fit gradient boosted trees.
                data.cascade[i].fit(rt);
                
                // Update shape estimate
                for (int s = 0; s < numSamples; ++s) {
                    t.samples[s].estimate +=
                        data.cascade[i].predict(t.input->images[t.samples[s].inputIdx],
                                                t.samples[s].estimate,
                                                t.samples[s].shapeToImage);
                }
            }
			
            // Update internal data
            data.meanShape = rt.meanShape;
			Rect shape_bounds = shapeBounds(data.meanShape);
			data.meanShapeRectCorners.resize(3, 4);
			data.meanShapeRectCorners(0, 0) = shape_bounds(0, 0);
			data.meanShapeRectCorners(0, 1) = shape_bounds(0, 1);
			data.meanShapeRectCorners(0, 2) = shape_bounds(0, 2);
			data.meanShapeRectCorners(0, 3) = shape_bounds(0, 3);
			data.meanShapeRectCorners(1, 0) = shape_bounds(1, 0);
			data.meanShapeRectCorners(1, 1) = shape_bounds(1, 1);
			data.meanShapeRectCorners(1, 2) = shape_bounds(1, 2);
			data.meanShapeRectCorners(1, 3) = shape_bounds(1, 3);
	
            return true;

        }
        
        Shape Tracker::predict(const Image &img, const ShapeTransform &shapeToImage, std::vector<Shape> *stepResults) const
        {

            Tracker::data &data = *_data;

			Shape estimate = data.meanShape;

			//**************************************************************
			/*
			typedef Eigen::Matrix<float, 2, 2> RotationMatrix;
			RotationMatrix r;
			r(0, 0) = cos(3.14f);
			r(0, 1) = -sin(3.14f);
			r(1, 0) = sin(3.14f);
			r(1, 1) = cos(3.14f);
			
			//旋转15°
			DEST_LOG("estimate" << estimate);
			//旋转15°
			std::cout << "estimate.colwise().homogeneous()" << estimate.colwise().homogeneous();

			estimate = r * estimate;


			//旋转15°
			std::cout << "estimate" << estimate << std::endl;
			//旋转15°
			std::cout << "estimate.colwise().homogeneous()" << estimate.colwise().homogeneous();
			*/
			//**************************************************************

            const int numCascades = static_cast<int>(data.cascade.size());
            for (int i = 0; i < numCascades; ++i) {
                if (stepResults) {
                    stepResults->push_back(shapeToImage * estimate.colwise().homogeneous());
                }
                estimate += data.cascade[i].predict(img, estimate, shapeToImage);
            }

            Shape final = shapeToImage * estimate.colwise().homogeneous();

            if (stepResults) {
                stepResults->push_back(final);
            }

            return final;
        }        
    }
}
