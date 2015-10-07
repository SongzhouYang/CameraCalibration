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
	ifstream fin("calibdata.txt"); /* ��������ͼ���ļ���·�� */
	ofstream fout("caliberation_result.txt");  /* ���涨�������ļ� */
	
    /************************************************************************
	       ��ȡÿһ��ͼ�񣬴�����ȡ���ǵ㣬Ȼ��Խǵ���������ؾ�ȷ��
	*************************************************************************/
	cout<<"��ʼ��ȡ�ǵ㡭����������";
	int image_count=0;  /* ͼ������ */
	CvSize image_size;  /* ͼ��ĳߴ� */
	CvSize board_size = cvSize(14,22);    /* �������ÿ�С��еĽǵ��� */
	CvPoint2D32f* image_points_buf = new CvPoint2D32f[board_size.width*board_size.height];   /* ����ÿ��ͼ���ϼ�⵽�Ľǵ� */
	Seq<CvPoint2D32f> image_points_seq;  /* �����⵽�����нǵ� */
	
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
		/* ��ȡ�ǵ� */
		if (0 == cvFindChessboardCorners( view.cvimage, board_size,
            image_points_buf, &count, CV_CALIB_CB_ADAPTIVE_THRESH ))
		{
			cout<<"can not find chessboard corners!\n";
			exit(1);
		} else {
			Image<uchar> view_gray(view.size(),8,1);
			rgb2gray(view,view_gray);
			/* �����ؾ�ȷ�� */
			cvFindCornerSubPix( view_gray.cvimage, image_points_buf, count, cvSize(11,11),
				cvSize(-1,-1), cvTermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
			image_points_seq.push_back(image_points_buf,count);
			/* ��ͼ������ʾ�ǵ�λ�� */
			cvDrawChessboardCorners( view.cvimage, board_size, image_points_buf, count, 1);
			view.show("calib");
			cvWaitKey();
			view.close();
		}
	}
	delete []image_points_buf;
	cout<<"�ǵ���ȡ��ɣ�\n";

	/************************************************************************
	       ���������
	*************************************************************************/
	cout<<"��ʼ���ꡭ����������";
	CvSize square_size = cvSize(10,10);  /* ʵ�ʲ����õ��Ķ������ÿ�����̸�Ĵ�С */
	Matrix<double> object_points(1,board_size.width*board_size.height*image_count,3); /* ���涨����Ͻǵ����ά���� */
	Matrix<double> image_points(1,image_points_seq.cvseq->total,2); /* ������ȡ�����нǵ� */
	Matrix<int> point_counts(1,image_count,1); /* ÿ��ͼ���нǵ������ */
	Matrix<double> intrinsic_matrix(3,3,1); /* ������ڲ������� */
	Matrix<double> distortion_coeffs(1,4,1); /* �������4������ϵ����k1,k2,p1,p2 */
	Matrix<double> rotation_vectors(1,image_count,3); /* ÿ��ͼ�����ת���� */
	Matrix<double> translation_vectors(1,image_count,3); /* ÿ��ͼ���ƽ������ */


	/* ��ʼ��������Ͻǵ����ά���� */
	int i,j,t;
	for (t=0;t<image_count;t++) {
		for (i=0;i<board_size.height;i++) {
			for (j=0;j<board_size.width;j++) {
				/* ���趨��������������ϵ��z=0��ƽ���� */
				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,0) = i*square_size.width;
				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,1) = j*square_size.height;
				object_points(0,t*board_size.height*board_size.width+i*board_size.width+j,2) = 0;
			}
		}
	}

	/* ���ǵ�Ĵ洢�ṹת���ɾ�����ʽ */
	for (i=0;i<image_points_seq.cvseq->total;i++) {
		image_points(0,i,0) = image_points_seq[i].x;
		image_points(0,i,1) = image_points_seq[i].y;
	}

	/* ��ʼ��ÿ��ͼ���еĽǵ��������������Ǽ���ÿ��ͼ���ж����Կ��������Ķ���� */
	for (i=0;i<image_count;i++)
		point_counts(0,i) = board_size.width*board_size.height;
	
	/* ��ʼ���� */
	cvCalibrateCamera2(object_points.cvmat,
					   image_points.cvmat,
                       point_counts.cvmat,
					   image_size,
                       intrinsic_matrix.cvmat,
					   distortion_coeffs.cvmat,
                       rotation_vectors.cvmat,
					   translation_vectors.cvmat,
					   0);
	cout<<"������ɣ�\n";
	
	/************************************************************************
	       �Զ�������������
	*************************************************************************/
	cout<<"��ʼ���۶�����������������\n";
	double total_err = 0.0; /* ����ͼ���ƽ�������ܺ� */
	double err = 0.0; /* ÿ��ͼ���ƽ����� */
	Matrix<double> image_points2(1,point_counts(0,0,0),2); /* �������¼���õ���ͶӰ�� */

	cout<<"\tÿ��ͼ��Ķ�����\n";
	fout<<"ÿ��ͼ��Ķ�����\n";
	for (i=0;i<image_count;i++) {
		/* ͨ���õ������������������Կռ����ά���������ͶӰ���㣬�õ��µ�ͶӰ�� */
		cvProjectPoints2(object_points.get_cols(i*point_counts(0,0,0),(i+1)*point_counts(0,0,0)-1).cvmat,
						rotation_vectors.get_col(i).cvmat,
						translation_vectors.get_col(i).cvmat,
						intrinsic_matrix.cvmat,
						distortion_coeffs.cvmat,
						image_points2.cvmat,
						0,0,0,0);
		/* �����µ�ͶӰ��;ɵ�ͶӰ��֮������*/
		err = cvNorm(image_points.get_cols(i*point_counts(0,0,0),(i+1)*point_counts(0,0,0)-1).cvmat,
					image_points2.cvmat,
					CV_L1);
		total_err += err/=point_counts(0,0,0);
		cout<<"\t\t��"<<i+1<<"��ͼ���ƽ����"<<err<<"����"<<'\n';
		fout<<"\t��"<<i+1<<"��ͼ���ƽ����"<<err<<"����"<<'\n';
	}
	cout<<"\t����ƽ����"<<total_err/image_count<<"����"<<'\n';
	fout<<"����ƽ����"<<total_err/image_count<<"����"<<'\n'<<'\n';
	cout<<"������ɣ�\n";

	/************************************************************************
	       ���涨����
	*************************************************************************/
	cout<<"��ʼ���涨����������������";
	Matrix<double> rotation_vector(3,1); /* ����ÿ��ͼ�����ת���� */
	Matrix<double> rotation_matrix(3,3); /* ����ÿ��ͼ�����ת���� */
	
	fout<<"����ڲ�������\n";
	fout<<intrinsic_matrix<<'\n';
	fout<<"����ϵ����\n";
	fout<<distortion_coeffs<<'\n';
	for (i=0;i<image_count;i++) {
		fout<<"��"<<i+1<<"��ͼ�����ת������\n";
		fout<<rotation_vectors.get_col(i);
		/* ����ת�������д洢��ʽת�� */
		for (j=0;j<3;j++) {
			rotation_vector(j,0,0) = rotation_vectors(0,i,j);
		}
		/* ����ת����ת��Ϊ���Ӧ����ת���� */
		cvRodrigues2(rotation_vector.cvmat,rotation_matrix.cvmat);
		fout<<"��"<<i+1<<"��ͼ�����ת����\n";
		fout<<rotation_matrix;
		fout<<"��"<<i+1<<"��ͼ���ƽ��������\n";
		fout<<translation_vectors.get_col(i)<<'\n';
	}
	cout<<"��ɱ���\n";
}