/**
    This file is part of Deformable Shape Tracking (DEST).

    Copyright(C) 2015/2016 Christoph Heindl
    All rights reserved.

    This software may be modified and distributed under the terms
    of the BSD license.See the LICENSE file for details.
*/

#include <dest/core/shape.h>
#include <Eigen/Dense>

namespace dest {
    namespace core {
        
        Eigen::AffineCompact3f estimateSimilarityTransform(const Eigen::Ref<const Shape> &from, const Eigen::Ref<const Shape> &to)
        {      

            Eigen::Vector3f meanFrom = from.rowwise().mean();
            Eigen::Vector3f meanTo = to.rowwise().mean();
            
            Shape centeredFrom = from.colwise() - meanFrom;
            Shape centeredTo = to.colwise() - meanTo;
            
            Eigen::Matrix3f cov = (centeredFrom) * (centeredTo).transpose();	//转置
            cov /= static_cast<float>(from.cols());
            const float sFrom = centeredFrom.squaredNorm() / from.cols();
            
            auto svd = cov.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix3f d = Eigen::Matrix3f::Zero(3, 3);
            d(0, 0) = svd.singularValues()(0);
			d(1, 1) = svd.singularValues()(1);
			d(2, 2) = svd.singularValues()(2);
            
            // Correct reflection if any.
            float detCov = cov.determinant();
            float detUV = svd.matrixU().determinant() * svd.matrixV().determinant();
            Eigen::Matrix3f s = Eigen::Matrix3f::Identity(3, 3);
			//判断是旋转矩阵还是反射矩阵：旋转矩阵的行列式值为+1，反射矩阵的行列值为-1
            if (detCov < 0.f || (detCov == 0.f && detUV < 0.f)) {
				if (svd.singularValues()(2) <= svd.singularValues()(0) && svd.singularValues()(2) <= svd.singularValues()(1)) {
                    s(2, 2) = -1;
                }
				else if (svd.singularValues()(1) <= svd.singularValues()(0) && svd.singularValues()(1) <= svd.singularValues()(2)) {
					s(1, 1) = -1;
				}
				else {
                    s(0, 0) = -1;
                }
            }
            
			//r是旋转矩阵
            Eigen::Matrix3f rot = svd.matrixU().transpose() * s * svd.matrixV();
            float c = 1.f;
            if (sFrom > 0) {
                c = 1.f / sFrom * (d * s).trace();
            }
            
			//t是偏移
            Eigen::Vector3f t = meanTo - c * rot * meanFrom;
            
            Eigen::Matrix<float, 3, 4> ret = Eigen::Matrix<float, 3, 4>::Identity(3, 4);
            ret.block<3,3>(0,0) = c * rot;
            ret.block<3,1>(0,3) = t;
            
            return Eigen::AffineCompact3f(ret);
        }
        
        int findClosestLandmarkIndex(const Shape &s, const Eigen::Ref<const Eigen::Vector3f> &x)
        {
            const int numLandmarks = static_cast<int>(s.cols());
            
            int bestLandmark = -1;
            float bestD2 = std::numeric_limits<float>::max();
            
            for (int i = 0; i < numLandmarks; ++i) {
                float d2 = (s.col(i) - x).squaredNorm();
                if (d2 < bestD2) {
                    bestD2 = d2;
                    bestLandmark = i;
                }
            }
            
            return bestLandmark;
        }
        
        
        void shapeRelativePixelCoordinates(const Shape &s, const PixelCoordinates &abscoords, PixelCoordinates &relcoords, Eigen::VectorXi &closestLandmarks)
        {
            
            relcoords.resize(abscoords.rows(), abscoords.cols());
            closestLandmarks.resize(abscoords.cols());
            
            const int numLocs = static_cast<int>(abscoords.cols());
            for (int i  = 0; i < numLocs; ++i) {
                int idx = findClosestLandmarkIndex(s, abscoords.col(i));
                relcoords.col(i) = abscoords.col(i) - s.col(idx);
                closestLandmarks(i) = idx;
            }
            
        }

        inline Rect getUnitRectangle() {
            Rect r(2, 4);

            // Top-left
            r(0, 0) = -0.5f;
            r(1, 0) = -0.5f;

            // Top-right
            r(0, 1) = 0.5f;
            r(1, 1) = -0.5f;

            // Bottom-left
            r(0, 2) = -0.5f;
            r(1, 2) = 0.5f;

            // Bottom-right
            r(0, 3) = 0.5f;
            r(1, 3) = 0.5f;

            return r;
        }

        const Rect &unitRectangle() {
            const static Rect _instance = getUnitRectangle();
            return _instance;
        }

        Rect shapeBounds(const Eigen::Ref<const Shape> &s)
        {
            const Eigen::Vector3f minC3f = s.rowwise().minCoeff();
            const Eigen::Vector3f maxC3f = s.rowwise().maxCoeff();

			const Eigen::Vector2f minC = Eigen::Vector2f(minC3f(0), minC3f(1));
			const Eigen::Vector2f maxC = Eigen::Vector2f(maxC3f(0), maxC3f(1));

			return createRectangle(minC, maxC);
        }

        Rect createRectangle(const Eigen::Vector2f &minC, const Eigen::Vector2f &maxC)
        {
            Rect rect(2, 4);
			rect.col(0) = minC;
            rect.col(1) = Eigen::Vector2f(maxC(0), minC(1));
            rect.col(2) = Eigen::Vector2f(minC(0), maxC(1));
			rect.col(3) = maxC;
            return rect;
        }
    }
}