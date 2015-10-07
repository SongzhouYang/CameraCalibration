#include "cvut.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace cvut;
using namespace std;

#pragma comment(lib,"cxcore.lib")
#pragma comment(lib,"cv.lib")
#pragma comment(lib,"highgui.lib")

void main() {
	ifstream fin("calibdata.txt"); /* 定标所用图像文件的路径 */
	ofstream fout("caliberation_result.txt");  /* 保存定标结果的文件 */
	
    /************************************************************************
	       读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
	*************************************************************************/
	cout<<"开始提取角点………………";
	int image_count=0;  /* 图像数量 */
	CvSize image_size;  /* 图像的尺寸 */
	CvSize board_size = cvSize(14,22);    /* 定标板上每行、列的角点数 */
	CvPoint2D32f* image_points_buf = new CvPoint2D32f[board_size.width*board_size.height];   /* 缓存每幅图像上检测到的角点 */
	Seq<CvPoint2D32f> image_points_seq;  /* 保存检测到的所有角点 */
	
	string filename;
	while (getline(fin,filename))
	{
		image_count++;
		int count;
		Image<uchar> view(filename);
		if (image_count == 1) {
			image_size.width = view.size().width;
			image_size.height = view.size().height;
		}
		/* 提取角点 */
		if (0 == cvFindChessboardCorners( view.cvimage, board_size,
            image_points_buf, &count, CV_CALIB_CB_ADAPTIVE_THRESH ))
		{
			cout<<"can not find chessboard corners!\n";
			exit(1);
		} else {
			Image<uchar> view_gray(view.size(),8,1);
			rgb2gray(view,view_gray);
			/* 亚像素精确化 */
			cvFindCornerSubPix( view_gray.cvimage, image_points_buf, count, cvSize(11,11),
				cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
			image_points_seq.push_back(image_points_buf,count);
			/* 在图像上显示角点位置 */
			cvDrawChessboardCorners( view.cvimage, board_size, image_points_buf, count, 1);
			view.show("calib");
			cvWaitKey();
			view.close();
		}
	}
	delete []image_points_buf;
	cout<<"角点提取完成！\n";

	/************************************************************************
	       摄像机定标
	*************************************************************************/
	cout<<"开始定标………………";
	CvSize square_size = cvSize(10,10);  /* 实际测量得到的定标板上每个棋盘格的大小 */
	Matrix<double> object_points(1,board_size.width*board_size.height*image_count,3); /* 保存定标板上角点的三维坐标 */
	Matrix<double> image_points(1,image_points_seq.cvseq->total,2); /* 保存提取的所有角点 */
	Matrix<int> point_counts(1,image_count,1); /* 每幅图像中角点的数量 */
	Matrix<double> intrinsic_matrix(3,3,1); /* 摄像机内参数矩阵 */
	Matrix<double> distortion_coeffs(1,4,1); /* 摄像机的4个畸变系数：k1,k2,p1,p2 */
	Matrix<double> rotation_vectors(1,image_count,3); /* 每幅图像的旋转向量 */
	Matrix<double> translation_vectors(1,image_count,3); /* 每幅图像的平移向量 */


	/* 初始化定标板上角点的三维坐标 */
	int i,j,t;
	for (t=0;t<image_count;t++) {
		for (i=0;i<board_size.height;i++) {
			for (j=0;j<board_size.width;j++) {
				/* 假设定标板放在世界坐标系中z=0的平面上 */
				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,0) = i*square_size.width;
				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,1) = j*square_size.height;
				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,2) = 0;
			}
		}
	}

	/* 将角点的存储结构转换成矩阵形式 */
	for (i=0;i<image_points_seq.cvseq->total;i++) {
		image_points(0,i,0) = image_points_seq[i].x;
		image_points(0,i,1) = image_points_seq[i].y;
	}

	/* 初始化每幅图像中的角点数量，这里我们假设每幅图像中都可以看到完整的定标板 */
	for (i=0;i<image_count;i++)
		point_counts(0,i) = board_size.width*board_size.height;
	
	/* 开始定标 */
	cvCalibrateCamera2(object_points.cvmat,
					   image_points.cvmat,
                       point_counts.cvmat,
					   image_size,
                       intrinsic_matrix.cvmat,
					   distortion_coeffs.cvmat,
                       rotation_vectors.cvmat,
					   translation_vectors.cvmat,
					   0);
	cout<<"定标完成！\n";
	
	/************************************************************************
	       对定标结果进行评价
	*************************************************************************/
	cout<<"开始评价定标结果………………\n";
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	Matrix<double> image_points2(1,point_counts(0,0,0),2); /* 保存重新计算得到的投影点 */

	cout<<"\t每幅图像的定标误差：\n";
	fout<<"每幅图像的定标误差：\n";
	for (i=0;i<image_count;i++) {
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		cvProjectPoints2(object_points.get_cols(i*point_counts(0,0,0),(i+1)*point_counts(0,0,0)-1).cvmat,
						rotation_vectors.get_col(i).cvmat,
						translation_vectors.get_col(i).cvmat,
						intrinsic_matrix.cvmat,
						distortion_coeffs.cvmat,
						image_points2.cvmat,
						0,0,0,0);
		/* 计算新的投影点和旧的投影点之间的误差*/
		err = cvNorm(image_points.get_cols(i*point_counts(0,0,0),(i+1)*point_counts(0,0,0)-1).cvmat,
					image_points2.cvmat,
					CV_L1);
		total_err += err/=point_counts(0,0,0);
		cout<<"\t\t第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<'\n';
		fout<<"\t第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<'\n';
	}
	cout<<"\t总体平均误差："<<total_err/image_count<<"像素"<<'\n';
	fout<<"总体平均误差："<<total_err/image_count<<"像素"<<'\n'<<'\n';
	cout<<"评价完成！\n";

	/************************************************************************
	       保存定标结果
	*************************************************************************/
	cout<<"开始保存定标结果………………";
	Matrix<double> rotation_vector(3,1); /* 保存每幅图像的旋转向量 */
	Matrix<double> rotation_matrix(3,3); /* 保存每幅图像的旋转矩阵 */
	
	fout<<"相机内参数矩阵：\n";
	fout<<intrinsic_matrix<<'\n';
	fout<<"畸变系数：\n";
	fout<<distortion_coeffs<<'\n';
	for (i=0;i<image_count;i++) {
		fout<<"第"<<i+1<<"幅图像的旋转向量：\n";
		fout<<rotation_vectors.get_col(i);
		/* 对旋转向量进行存储格式转换 */
		for (j=0;j<3;j++) {
			rotation_vector(j,0,0) = rotation_vectors(0,i,j);
		}
		/* 将旋转向量转换为相对应的旋转矩阵 */
		cvRodrigues2(rotation_vector.cvmat,rotation_matrix.cvmat);
		fout<<"第"<<i+1<<"幅图像的旋转矩阵：\n";
		fout<<rotation_matrix;
		fout<<"第"<<i+1<<"幅图像的平移向量：\n";
		fout<<translation_vectors.get_col(i)<<'\n';
	}
	cout<<"完成保存\n";
}