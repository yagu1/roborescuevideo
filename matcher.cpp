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
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;
void readme();
void search_dir(string path,  vector<string> &fileNames);
class ImageData{
public:
  string path;
  Mat image;
  Mat gray;
  vector<cv::KeyPoint> keys;
  Mat desc;
};

int main(int argc, char** argv)
{
  //if(argc != 3){ readme(); return -1;}
  VideoCapture cap(0);//デバイスのオープン
  cap.set(CAP_PROP_FRAME_WIDTH, 1280);
  cap.set(CAP_PROP_FRAME_HEIGHT,720);

  int i;
  vector<string> hoge;
  hoge.clear();

  search_dir("./images/", hoge);

  for(i=0;i<hoge.size();i+=1){
    cout << hoge[i] << endl;
  }
  vector<ImageData> imgdts;
  cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f);
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
  int dictionarySize = 122;
  TermCriteria tc(TermCriteria::MAX_ITER, 100, 0.001);
  int retries = 1;
  int flags = KMEANS_PP_CENTERS;
  BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);//BOW_kmeans設定
  BOWImgDescriptorExtractor bowDE(akaze, matcher);//BOW_を実際に型の指定，(AKAZE,FlannBaed)

  for(i=0;i<hoge.size();i+=1){
    ImageData imgdt;
    Mat img = imread(hoge[i],-1);
    Mat image;

    cv::resize(img,image, cv::Size(), 1000.0/img.rows, 1000.0/img.cols);
    imgdt.path = hoge[i];
    imgdt.image = image;
    cvtColor(imgdt.image, imgdt.gray, COLOR_BGR2GRAY);
    //waitKey(1000);
    try{
      cv::Mat dstAkaze;
      cv::Mat descorg;
      // 検出したキーポイント（特徴点）を格納する配列
      std::vector<cv::KeyPoint> keyAkaze;
      cv::Mat desc;
      akaze->detectAndCompute(imgdt.gray, cv::Mat(),imgdt.keys, descorg);
      descorg.convertTo(imgdt.desc, CV_32FC1);
      // 画像上にキーポイントの場所を描く
      // # DrawMatchesFlags::DRAW_RICH_KEYPOINTS  キーポイントのサイズと方向を描く
      cv::drawKeypoints(imgdt.image, imgdt.keys, dstAkaze, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      cv::imshow("AKAZE", dstAkaze);

      const int key = waitKey(1);
      if(key == 'q' || key == 27/*113*/)//qボタンが押されたとき
      {
        break;//whileループから抜ける．
      }
      else if(key == 's'/*115*/)//sが押されたとき
      {
        //フレーム画像を保存する．
        cv::imwrite("img.png", image);
      }
    }catch(int fError){
      cout<< fError << endl;
    }
    bowTrainer.add(imgdt.desc);
    imgdts.push_back(imgdt);
  }
  Mat codebook = bowTrainer.cluster();

  Ptr<BackgroundSubtractor> pBackSub;
  pBackSub = cv::createBackgroundSubtractorMOG2();
  if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
  {
    cout << "Ne-yo" << endl;
      //読み込みに失敗したときの処理
      return -1;
  }
  bowDE.setVocabulary(codebook);
  while(true)//無限ループ
  {
    Mat frame, gray, fgMask, desc, descorg;
    std::vector<cv::KeyPoint> keys;
    Mat imgDesc;
    cap.read(frame);
    cvtColor(frame,gray,COLOR_BGR2GRAY);

    // 検出したキーポイント（特徴点）を格納する配列
    akaze->detectAndCompute(gray, cv::Mat(), keys, descorg);
    descorg.convertTo(desc,CV_32FC1);
    vector< vector< int > >* pIdxC;
    //bowDE.compute(desc,imgDesc,pIdxC);

    //cout << cv::format(imgDesc,Formatter::FMT_CSV);
    cv::imshow("current", frame);
    cout << imgdts.size() << endl;
    for(i=0;i<imgdts.size();i++){
      cout << i << endl;
      ImageData imgdt = imgdts[i];
      vector< vector< cv::DMatch > > knn_matches;
      matcher->knnMatch(desc,imgdt.desc,knn_matches,2);

      const auto match_par = .8f; //対応点のしきい値
      std::vector<cv::DMatch> good_matches;
      std::vector<cv::Point2f> match_point1;
      std::vector<cv::Point2f> match_point2;

      for (size_t i = 0; i < knn_matches.size(); ++i) {
        auto dist1 = knn_matches[i][0].distance;
        auto dist2 = knn_matches[i][1].distance;

        //良い点を残す（最も類似する点と次に類似する点の類似度から）
        if (dist1 <= dist2 * match_par) {
          good_matches.push_back(knn_matches[i][0]);
          match_point1.push_back(keys[knn_matches[i][0].queryIdx].pt);
          match_point2.push_back(imgdt.keys[knn_matches[i][0].trainIdx].pt);
        }
      }
      //ホモグラフィ行列推定
      cv::Mat masks;
      cv::Mat H;
      cerr << "b";
      if (match_point1.size() != 0 && match_point2.size() != 0) {
        cerr << "c";
          H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC, 3.f);
      }

      //RANSACで使われた対応点のみ抽出
      std::vector<cv::DMatch> inlierMatches;
      for (auto i = 0; i < masks.rows; ++i) {
        uchar *inlier = masks.ptr<uchar>(i);
        if (inlier[0] == 1) {
            inlierMatches.push_back(good_matches[i]);
        }
      }/**/
      Mat dst;
      cout << "good" << inlierMatches.size() << endl;
      if(inlierMatches.size() < 10){
        continue;
      }
      cv::drawMatches(frame,keys , imgdts[i].image, imgdts[i].keys, inlierMatches, dst);
      //インライアの対応点のみ表示
      imshow("d",dst);
      const int key = waitKey(100);
      /**/
    }
    const int key = waitKey(1);
    if(key == 'q' || key == 27/*113*/)//qボタンが押されたとき
    {
      break;//whileループから抜ける．
    }
    else if(key == 's'/*115*/)//sが押されたとき
    {
      //フレーム画像を保存する．
      cv::imwrite("img.png", frame);
    }
  }
  return 0;

}

void search_dir(string path,  vector<string> &fileNames){

  int i, dirElements;
  string search_path;

  struct stat stat_buf;
  struct dirent **namelist=NULL;

  // dirElements にはディレクトリ内の要素数が入る
  dirElements = scandir(path.c_str(), &namelist, NULL, NULL);

  if(dirElements == -1) {
    cout << "ERROR" <<  endl;
  }

  else{

    //ディレクトリかファイルかを順番に識別
    for (i=0; i<dirElements; i+=1) {

      // "." と ".." を除く
      if( (strcmp(namelist[i]->d_name , ".\0") != 0) && (strcmp(namelist[i]->d_name , "..\0") != 0) ){

        //search_pathには検索対象のフルパスを格納する
        search_path = path + string(namelist[i] -> d_name);

        // ファイル情報の取得の成功
        if(stat(search_path.c_str(), &stat_buf) == 0){

          // ディレクトリだった場合の処理
          if ((stat_buf.st_mode & S_IFMT) == S_IFDIR){
            // 再帰によりディレクトリ内を探索
            search_dir(path + string(namelist[i] -> d_name) + "/", fileNames);
          }

          //ファイルだった場合の処理
          else {
            fileNames.push_back(search_path);
          }
        }

        // ファイル情報の取得の失敗
        else{
          cout << "ERROR" <<  endl << endl;
        }
      }
    }
  }

  free(namelist);
  return;
}
