#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/bgsegm.hpp>

using namespace cv;
using namespace std;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

int main(){

  //AKAZE as feature extractor and FlannBased for assigning a new point to the nearest one in the dictionary

  auto algo = AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(“FlannBased”);
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::craete("AKAZE");

  int dictionarySize = 200;//辞書サイズ1500
  TermCriteria tc(CV_TERMCRIT_ITER, 10, 0.001);
  int retries = 1;
  int flags = KMEANS_PP_CENTERS;
  BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);//BOW_kmeans設定
  BOWImgDescriptorExtractor bowDE(extractor, matcher);//BOW_を実際に型の指定，(SURF,FlannBaed)
  const int CLASS =2;
  const int MAISU=5;
  std::stringstream name;
  int i,j;
  for(int i=0;i<CLASS;i++)
  for(int j=0;j<MAISU;j++){
    name.str(“”);
    name<<i<<“/”<<j<<“.jpg”;
    cv::Mat img=cv::imread(name.str(),0);
    vector<KeyPoint> keypoint;
    Mat features;
    algo->detect(img, keypoint);
    algo->compute(img, keypoint, features);
    bowTrainer.add(features);//ここに追加していく．BOWKMeansTrainer型の変数
  }
  //辞書の作成
  Mat dictionary = bowTrainer.cluster();
  //辞書のファイルへの保存
  FileStorage cvfs(“test_2.xml”, CV_STORAGE_WRITE);
  write(cvfs,“VOB”, dictionary);
  cvfs.release();

  ///////↑で辞書データの作成はできている．
  //ここからは辞書データの読み込みを説明するために，あえて別の変数を用意している．
  Mat dictionary_read;
  FileStorage cvfread(“test_2.xml”, CV_STORAGE_READ);
  cvfread[“VOB”]>>dictionary_read;
  bowDE.setVocabulary(dictionary_read);//これが辞書データを用いて，特徴量の置き換えを行う．　
  cv::Mat test_img=cv::imread(“0/1.jpg”,0);
  cv::Mat test_bowDescriptor;
  vector<KeyPoint> test_keypoint;
  detector.detect(test_img, test_keypoint);
  bowDE.compute(test_img, test_keypoint, test_bowDescriptor);
  float sum=0;
  for(int i=0;i<test_bowDescriptor.rows;i++)
  for(int j=0;j<test_bowDescriptor.cols;j++)
  sum+=test_bowDescriptor.at<float>(i,j);
  std::cout<<test_bowDescriptor<<“\n”;
  // std::cout<<“test_bowDescriptorの次元数:”<<test_bowDescriptor.rows*test_bowDescriptor.cols;
  std::cout<<“↑test_bowDescriptorの合計:”<<sum<<“\n”;
  getchar();
  ｝
