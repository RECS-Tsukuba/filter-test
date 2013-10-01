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
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>

using std::atof;
using std::bind;
using std::count;
using std::cout;
using std::cerr;
using std::endl;
using std::exception;
using std::getline;
using std::ifstream;
using std::ios;
using std::placeholders::_1;
using std::string;
using std::stringstream;
using cv::filter2D;
using cv::imread;
using cv::Mat;
using cv::namedWindow;
using cv::Rect;
using cv::waitKey;

namespace {

template <typename F, typename G>
class KleisliCompositedMatFunctor {
 private:
  F f_;
  G g_;
 public:
  KleisliCompositedMatFunctor(F f, G g): f_(f), g_(g) {}
 public:
  template <typename... Args>
  inline cv::Mat operator()(Args&&... args) {
    cv::Mat r = f_(std::forward<Args>(args)...);
    return (r.empty())? r: g_(r);
  }
};

template <typename F, typename G>
constexpr KleisliCompositedMatFunctor<F, G> operator>=(F f, G g) noexcept
  { return KleisliCompositedMatFunctor<F, G>(f, g); }

}  // namespace

namespace {
cv::Mat Conbime(Mat output, const cv::Mat& left, const cv::Mat& right);
cv::Mat Filter(const cv::Mat& src, const cv::Mat& kernel);
int GetExitCode(cv::Mat m) noexcept;
std::string GetImageFilename(int argc, char** argv);
cv::Mat GetKernel(const std::string& filename);
std::string GetKernelFilename(int argc, char** argv);
int GetKernelSize(std::ifstream& stream);
int Help() noexcept;
cv::Mat MakeOutputImage(const cv::Mat& src);
std::ifstream& Rewind(std::ifstream& stream);
void SetOperator(cv::Mat row, const string& line, int size);
cv::Mat SetOperators(cv::Mat kernel, std::ifstream& stream, int size);
Mat Show(cv::Mat image);

}  // namespace

namespace {
/*!
 \brief 二つの画像を水平に結合する。
 出力先である'output'は事前に確保する必要がある。
 'left'及び'right'の画像サイズは'output'の画像サイズの半分以下である必要がある。

 \param output 出力先画像
 \param left 左画像
 \param right 右画像
 \return 'output'
 */
Mat Combine(Mat output, const Mat& left, const Mat& right) {
  left.copyTo(Mat(output, Rect(0, 0, left.cols, left.rows)));
  right.copyTo(Mat(output, Rect(left.cols, 0, right.cols, right.rows)));
  return output;
}
/*!
 \brief 画像をカーネルに基づいた線形フィルタをかける。

 \param original 元画像
 \param kernel カーネル
 \return フィルタされた画像。元画像かカーネルにエラーがある場合、空の画像
*/
Mat Filter(const Mat& original, const Mat& kernel) {
  Mat filtered;
  original.copyTo(filtered);

  filter2D(original, filtered, original.depth(), kernel, cv::Point(0, 0));
  return filtered;
}
/*!
 \brief 画像の状態に応じた終了コードを取得

 \param m 画像
 \return 'm'が正常ならばEXIT_SUCCESS、'm'にエラーがあるならばEXIT_FAILURE。
 */
int GetExitCode(cv::Mat m) noexcept
  { return (m.empty())? EXIT_FAILURE: EXIT_SUCCESS; }
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
  ifstream stream(filename.c_str());
  if (stream.good()) {
    int size = ::GetKernelSize(stream);
    // カーネルの領域を確保
    Mat kernel = Mat::zeros(size, size, cv::DataType<double>::type);

    return (size <= 0 || kernel.empty())?
      Mat(): SetOperators(kernel, ::Rewind(stream), size);
  } else { return Mat(); }
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
 \brief ヘルプを表示。

 \return 常にEXIT_SUCCESS
 */
int Help() noexcept {
  cerr <<"Usage: linear_filter filter_csv [image_file]" << std::endl;
  return EXIT_SUCCESS;
}
/*!
 \brief 出力先の画像を生成。
 元画像の二倍の大きさの、空の画像を生成する。

 \param src 元画像
 \return 出力先画像
 */
Mat MakeOutputImage(const Mat& src) {
  return Mat::zeros(src.size().height, src.size().width * 2, src.type());
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
 \brief 画像をウィンドウへ表示。

 \param image 画像
 \return 'image'
*/
Mat Show(Mat image) {
  string window_name("linear_filter");
  namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  imshow(window_name, image);

  waitKey(0);

  return image;
}

}  // namespace

namespace {
auto get_image = [](int argc, char** argv) {
  return imread(::GetImageFilename(argc, argv), CV_LOAD_IMAGE_GRAYSCALE);
};

auto get_kernel = [](int argc, char** argv)
  { return ::GetKernel(::GetKernelFilename(argc, argv)); };

auto show = [](Mat src, Mat filtered) {
  auto show_detail = ::MakeOutputImage >=
    bind(Combine, _1, src, filtered) >= ::Show;
  return show_detail(src);
};

auto test_filter = [](int argc, char** argv) {
  auto filter_tester = get_image >=
    [argc, argv](Mat src) {
      auto filter_and_show = get_kernel >= bind(Filter, src, _1) >=
        bind(show, src, _1);
      return filter_and_show(argc, argv);
    };

  return filter_tester(argc, argv);
};

}  // namespace

int main(int argc, char** argv) {
  try {
    exit((argc >= 2)?  ::GetExitCode(::test_filter(argc, argv)): ::Help());
  } catch(const exception& e) {
    cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  } catch(...) {
    cerr << "failed to filter the image" << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}

