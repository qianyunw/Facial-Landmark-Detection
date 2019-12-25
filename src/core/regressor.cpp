/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/regressor.h>
#include <dest/core/tree.h>
#include <dest/util/log.h>
#include <dest/io/dest_io_generated.h>
#include <dest/io/matrix_io.h>

namespace dest {
    namespace core {
        
        struct Regressor::data {
            
            PixelCoordinates shapeRelativePixelCoordinates;
            Eigen::VectorXi closestShapeLandmark;
            
            ShapeResidual meanResidual;
            Shape meanShape;
            std::vector<Tree> trees;
            float learningRate;
            
            data()
            {}

            flatbuffers::Offset<io::Regressor> save(flatbuffers::FlatBufferBuilder &fbb) const {
                flatbuffers::Offset<io::MatrixF> lpixels = io::toFbs(fbb, shapeRelativePixelCoordinates);
                flatbuffers::Offset<io::MatrixI> lcosest = io::toFbs(fbb, closestShapeLandmark);
                flatbuffers::Offset<io::MatrixF> lmeanr = io::toFbs(fbb, meanResidual);
                flatbuffers::Offset<io::MatrixF> lmeans = io::toFbs(fbb, meanShape);
                

                std::vector< flatbuffers::Offset<io::Tree> > ltrees;
                for (size_t i = 0; i < trees.size(); ++i) {
                    ltrees.push_back(trees[i].save(fbb));
                }
                auto vtrees = fbb.CreateVector(ltrees);

                io::RegressorBuilder b(fbb);
                b.add_closestLandmarks(lcosest);
                b.add_pixelCoordinates(lpixels);
                b.add_meanShapeResidual(lmeanr);
                b.add_meanShape(lmeans);
                b.add_forest(vtrees);
                b.add_learningRate(learningRate);

                return b.Finish();
            }

            void load(const io::Regressor &fbs) {

                io::fromFbs(*fbs.closestLandmarks(), closestShapeLandmark);
                io::fromFbs(*fbs.pixelCoordinates(), shapeRelativePixelCoordinates);
                io::fromFbs(*fbs.meanShapeResidual(), meanResidual);
                io::fromFbs(*fbs.meanShape(), meanShape);
                learningRate = fbs.learningRate();

                trees.resize(fbs.forest()->size());
                for (flatbuffers::uoffset_t i = 0; i < fbs.forest()->size(); ++i) {
                    trees[i].load(*fbs.forest()->Get(i));
                }
            }


        };
        
        Regressor::Regressor()
        : _data(new data())
        {
        }
        
        Regressor::Regressor(const Regressor &other)
        :_data(new data(*other._data))
        {}
        
        Regressor::~Regressor()
        {}

        flatbuffers::Offset<io::Regressor> Regressor::save(flatbuffers::FlatBufferBuilder &fbb) const {
            return _data->save(fbb);
        }

        void Regressor::load(const io::Regressor &fbs) {
            _data->load(fbs);
        }
        
        bool Regressor::fit(RegressorTraining &t)
        {
            Regressor::data &data = *_data;
            SampleData &tdata = *t.training;

            data.learningRate = t.training->params.learningRate;
            data.trees.resize(t.training->params.numTrees);
            data.meanShape = t.meanShape;
            
            TreeTraining tt;
            tt.numLandmarks = t.numLandmarks;
            tt.training = t.training;
            tt.input = t.input;
            tt.samples.resize(t.training->samples.size());
            
            // Draw random samples
            tt.pixelCoordinates = sampleCoordinates(t);
            
            // Encode them with respect to the mean shape
            shapeRelativePixelCoordinates(t.meanShape, tt.pixelCoordinates, data.shapeRelativePixelCoordinates, data.closestShapeLandmark);
            
            // Compute the mean residual, to be used as base learner
            data.meanResidual = ShapeResidual::Zero(3, t.numLandmarks);
            for (size_t i = 0; i < tdata.samples.size(); ++i) {

                tt.samples[i].residual = tdata.samples[i].target - tdata.samples[i].estimate;
                data.meanResidual += tt.samples[i].residual;
                
                Eigen::AffineCompact3f tShapeToShape = estimateSimilarityTransform(t.meanShape, tdata.samples[i].estimate);
                Eigen::AffineCompact3f tShapeToImage = tdata.samples[i].shapeToImage;

                readPixelIntensities(tShapeToShape,
                                     tShapeToImage,
                                     tdata.samples[i].estimate,
                                     t.input->images[tdata.samples[i].inputIdx],
                                     tt.samples[i].intensities);
                
            }
            data.meanResidual /= static_cast<float>(tdata.samples.size());
            
			//������
            for (int k = 0; k < t.training->params.numTrees; ++k) {
				DEST_LOG("Building tree " << std::setw(5) << k + 1 << "\r");
                for (size_t i = 0; i < tdata.samples.size(); ++i) {
                    
                    if (k == 0) {
                        tt.samples[i].residual -= data.meanResidual;
                    } else {
                        tt.samples[i].residual -= data.learningRate * data.trees[k - 1].predict(tt.samples[i].intensities);
                    }
                }
                data.trees[k].fit(tt);
            }
            
            
            
            return false;
        }
        
        PixelCoordinates Regressor::sampleCoordinates(RegressorTraining &t) const {
            
            Eigen::Vector3f minC = t.meanShape.rowwise().minCoeff() - Eigen::Vector3f::Constant(t.training->params.expansionRandomPixelCoordinates);
            Eigen::Vector3f maxC = t.meanShape.rowwise().maxCoeff() + Eigen::Vector3f::Constant(t.training->params.expansionRandomPixelCoordinates);

            const int numCoords = t.training->params.numRandomPixelCoordinates;
            PixelCoordinates result(3, numCoords);
            
            std::uniform_real_distribution<float> dx(0.f, maxC.x() - minC.x());
            std::uniform_real_distribution<float> dy(0.f, maxC.y() - minC.y());
			std::uniform_real_distribution<float> dz(0.f, maxC.z() - minC.z());

            for (int i = 0; i < numCoords; ++i) {
                result(0, i) = minC.x() + dx(t.input->rnd);
				result(1, i) = minC.y() + dy(t.input->rnd);
				result(2, i) = minC.z() + dz(t.input->rnd);
            }
            
            return result;
        }
        
        
        void Regressor::readPixelIntensities(const Eigen::AffineCompact3f &shapeToShape, const Eigen::AffineCompact3f &shapeToImage, const Shape &s, const Image &img, PixelIntensities &intensities) const
        {
            Regressor::data &data = *_data;
            
            PixelCoordinates coords = shapeToShape.matrix().block<3,3>(0,0) * data.shapeRelativePixelCoordinates;
            
            const Shape::Index numCoords = data.shapeRelativePixelCoordinates.cols();
            for(Shape::Index i = 0; i < numCoords; ++i) {
                coords.col(i) += s.col(data.closestShapeLandmark(i));
            }
            
            coords = shapeToImage.matrix() * coords.colwise().homogeneous();

            readImage(img, coords, intensities);
        }
        
        ShapeResidual Regressor::predict(const Image &img, const Shape &shape, const ShapeTransform &shapeToImage) const
        {
            Regressor::data &data = *_data;
            
            PixelIntensities intensities;
            Eigen::AffineCompact3f shapeToShape = estimateSimilarityTransform(data.meanShape, shape);
            readPixelIntensities(shapeToShape, shapeToImage, shape, img, intensities);
            
            const size_t numTrees = data.trees.size();
            
            ShapeResidual sr = data.meanResidual;
            for(size_t i = 0; i < numTrees; ++i) {
                sr += data.trees[i].predict(intensities) * data.learningRate;
            }
            
            return sr;
        }
    }
}