/*
 Copyright (c) 2013
 Reconfigurable computing systems laboratory, University of Tsukuba
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using std::atof;
using std::count;
using std::cout;
using std::cerr;
using std::endl;
using std::getline;
using std::ifstream;
using std::ios;
using std::string;
using std::stringstream;
using cv::filter2D;
using cv::imread;
using cv::Mat;
using cv::namedWindow;
using cv::Rect;
using cv::waitKey;

namespace {
cv::Mat Filter(const cv::Mat& src, const cv::Mat& kernel);
std::string GetImageFilename(int argc, char** argv);
cv::Mat GetKernel(const std::string& filename);
std::string GetKernelFilename(int argc, char** argv);
int GetKernelSize(std::ifstream& stream);
std::ifstream& Rewind(std::ifstream& stream);
void SetOperator(cv::Mat row, const string& line, int size);
cv::Mat SetOperators(cv::Mat kernel, std::ifstream& stream, int size);
int ShowErrorWindow(const std::string& error_message);
int ShowImageWindow(const cv::Mat& original, const cv::Mat& filtered);
int ShowWindow(const cv::Mat& original, const cv::Mat& filtered);
}  // namespace

namespace {
/*!
 \brief 画像をカーネルに基づいた線形フィルタをかける。

 \param src 元画像
 \param kernel カーネル
 \return フィルタされた画像。元画像かカーネルにエラーがある場合、空の画像
*/
Mat Filter(const Mat& src, const Mat& kernel) {
  if (!src.empty() && !kernel.empty()) {
    Mat filtered;
    src.copyTo(filtered);

    filter2D(src, filtered, src.depth(), kernel, cv::Point(0, 0));
    return filtered;
  } else { return Mat(); }
}
/*!
 \brief 画像のファイル名を取得。

 プログラム引数が2つの時(画像ファイル名が指定されていない)はデフォルトの値
 'input.jpg'を、それ以上の場合は第二引数をファイル名として返す。
 
 \param agrc argc
 \param argv argv
 \return 画像のファイル名
*/
string GetImageFilename(int argc, char** argv)
  { return (argc == 2)? string("input.jpg"): string(argv[2]); }
/*!
 \brief ファイルを読み込み、カーネルを取得する。

 \param filename カーネルを表したCSVのファイル名
 \return カーネル。エラーが起きた場合は空の画像
*/
Mat GetKernel(const string& filename) {
  try {
    ifstream stream(filename.c_str());
    if (stream.good()) {
      int size = ::GetKernelSize(stream);
      // カーネルの領域を確保
      Mat kernel = Mat::zeros(size, size, cv::DataType<double>::type);

      return (size <= 0 || kernel.empty())?
        Mat(): SetOperators(kernel, ::Rewind(stream), size);
    } else { return Mat(); }
  } catch(...) { return Mat(); }
}
/*!
 \brief カーネルを記述したファイル名を取得。

 \param argc argc
 \param argv argv
 \return カーネルを記述したファイル名
 */
string GetKernelFilename(int argc, char** argv) { return string(argv[1]); }
/*!
 \brief カーネルのサイズを取得。

 \param stream カーネルのファイルストリーム
 \return カーネルのサイズ。取得に失敗した場合は0
 */
int GetKernelSize(ifstream& stream) {
  string line;
  return (getline(stream, line))?
    count(line.begin(), line.end(), ',') + 1: 0;
}
/*!
 \brief ファイルストリームの走査位置を先頭へ戻す。

 \param stream ファイルストリーム
 \return ファイルストリーム
 */
ifstream& Rewind(ifstream& stream) {
  stream.clear();
  stream.seekg(0, ios::beg);
  return stream;
}
/*!
 \brief カーネルへオペレータをセット。

 \param row カーネルの行
 \param line ファイルから読み込まれた文字列
 \param size カーネルのサイズ
*/
void SetOperator(Mat row, const string& line, int size) {
  stringstream line_stream(line);
  string op;
  for (int i = 0; i < size && getline(line_stream, op, ','); ++i)
    { row.at<double>(i) = static_cast<double>(atof(op.c_str())); }
}
/*!
 \brief カーネルの全ての要素へオペレータをセット。

 \param kernel カーネル
 \param stream ファイルストリーム
 \param size カーネルのサイズ
 \return カーネル
 */
Mat SetOperators(Mat kernel, ifstream& stream, int size) {
  string line;
  for (int i = 0; i < size && getline(stream, line); ++i)
    { SetOperator(kernel.row(i), line, size); }

  return kernel;
}
/*!
 \brief エラーメッセージをウィンドウ名として表示。

 ウィンドウが閉じられた時、標準エラー出力にもエラーメッセージを出力する。

 \TODO もっといいエラーの表示方法はないものか...

 \param error_message エラーメッセージ
 \return 常ににEXIT_FAILURE
*/
int ShowErrorWindow(const string& error_message) {
  namedWindow(error_message, CV_WINDOW_AUTOSIZE);
  waitKey(0);

  cerr << error_message << endl;

  return EXIT_FAILURE;
}
/*!
 \brief 元画像をフィルタされた画像を表示。

 \param original 元画像
 \param filtered フィルタされた画像
 \return 常にEXIT_SUCCESS
*/
int ShowImageWindow(const Mat& original, const Mat& filtered) {
  Mat output = Mat::zeros(
      original.size().height,
      original.size().width * 2,
      original.type());

  if (output.empty()) {
    return ShowErrorWindow("failed to allocate an output image");
  } else {
    // 元画像とフィルタされた画像を結合
    original.copyTo(Mat(output, Rect(0, 0, original.cols, original.rows)));
    filtered.copyTo(
        Mat(output, Rect(original.cols, 0, original.cols, original.rows)));

    string window_name("linear_filter");
    namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    imshow(window_name, output);

    waitKey(0);

    return EXIT_SUCCESS;
  }
}
/*!
 \brief ウィンドウを表示し、結果を出力。

 　'original'か'filtered'にエラーがある場合、エラーウィンドウを表示する。それ以
 外の場合はフィルタされた画像を表示する。

 \param original 元画像
 \param filtered フィルタされた画像
 \param エラーコード
 */
int ShowWindow(const Mat& original, const Mat& filtered) {
  if (original.empty()) {
    return ShowErrorWindow(string("failed to open the image."));
  } else if (filtered.empty()) {
    return ShowErrorWindow(string("failed to filter the image."));
  } else { return ShowImageWindow(original, filtered); }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 || argc == 3) {
    Mat original = imread(::GetImageFilename(argc, argv),
                          CV_LOAD_IMAGE_GRAYSCALE);

    exit(
        ::ShowWindow(
            original,
            ::Filter(original, ::GetKernel(::GetKernelFilename(argc, argv)))));
  } else {
    exit(
        ::ShowErrorWindow(
            string("Usage: linear_filter filter_csv [image_file]")));
  }

  return 0;
}

