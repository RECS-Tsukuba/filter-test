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
// 2013/10現在、標準ライブラリの<regex>は実装されていないため、boost
// ライブラリを使用する
#include <boost/regex.hpp>
//#include <regex>
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
using std::placeholders::_2;
using std::ref;
// 2013/10現在、標準ライブラリの<regex>は実装されていないため、boost
// ライブラリを使用する
//using std::regex;
//using std::regex_match;
//using std::regex_constants::extended;
using std::string;
using std::stringstream;
using cv::filter2D;
using cv::imread;
using cv::Mat;
using cv::namedWindow;
using cv::Rect;
using cv::waitKey;
using boost::regex;
using boost::regex_match;
using boost::regex_constants::extended;

namespace {
/*!
 \brief Kleisli合成された関数オブジェクト

 (a -> m b) -> (b -> m c) -> a -> m c
 */
template <typename F, typename G>
class KleisliCompositedMatFunctor {
 private:
  F f_;
  G g_;
 public:
  /*!
   \brief Kleisli合成された関数オブジェクトを生成する

   \param f cv::Matを引数とする関数
   \param g cv::Matを引数と戻り値とする関数
   */
  KleisliCompositedMatFunctor(F f, G g): f_(f), g_(g) {}
 public:
  /*!
   \brief Kleisli合成された関数の実体

   \param args fの引数
   \return gの戻り値
   */
  template <typename... Args>
  inline cv::Mat operator()(Args&&... args) {
    cv::Mat r = f_(std::forward<Args>(args)...);
    return (r.empty())? r: g_(r);
  }
};
/*!
 \brief 二つの関数に対してKleisli合成を行う。

 \param f cv::Matを引数とする関数
 \param g cv::Matを引数と戻り値とする関数
 \return Kleisli合成された関数オブジェクト
 */
template <typename F, typename G>
constexpr KleisliCompositedMatFunctor<F, G> operator>=(F f, G g) noexcept
  { return KleisliCompositedMatFunctor<F, G>(f, g); }
}  // namespace

namespace {
bool CheckKernelLine(const std::string& line);
cv::Mat Conbime(Mat output, const cv::Mat& left, const cv::Mat& right);
cv::Mat Filter(const cv::Mat& src, const cv::Mat& kernel);
int GetExitCode(cv::Mat m) noexcept;
std::string GetImageFilename(int argc, char** argv);
cv::Mat GetKernel(const std::string& filename);
std::string GetKernelFilename(int argc, char** argv);
int GetKernelSize(std::ifstream& stream);
cv::Mat HandleEmpty(cv::Mat m, const char* const error_message);
int Help() noexcept;
cv::Mat MakeKernel(int size);
cv::Mat MakeOutputImage(const cv::Mat& src);
void ExitFailure(const char* const error_message);
std::ifstream& Rewind(std::ifstream& stream);
void SetOperator(cv::Mat row, const string& line, int size);
cv::Mat SetOperators(cv::Mat kernel, std::ifstream& stream);
cv::Mat SetOperators_impl(cv::Mat kernel, std::ifstream& stream);
Mat Show(cv::Mat image);
}  // namespace

namespace {
/*!
 \brief 文字列がカーネルの一行を表しているか判定する

 \param line 文字列
 \return カーネルの一行を示している場合、true。それ以外の場合false
 */
bool CheckKernelLine(const string& line)
  { return regex_match(line, regex("(-*[0-9.]+,)*(-*[0-9.]+)", extended)); }
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
  auto impl = ::MakeKernel >= bind(SetOperators, _1, ref(stream));
  return (stream.good())? impl(::GetKernelSize(stream)): Mat();
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
  return (getline(stream, line) && ::CheckKernelLine(line))?
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
 \brief 空の画像だった場合、エラーメッセージを表示し終了する

 \param m 画像
 \param error_message エラーメッセージ 
 \return m
 */
Mat HandleEmpty(Mat m, const char* const error_message) {
  if (m.empty()) { ::ExitFailure(error_message); }
  return m;
}
Mat MakeKernel(int size) {
  return (size == 0)?
    Mat():Mat::zeros(size, size, cv::DataType<double>::type);
}
/*!
 \brief 出力先の画像を生成。
 元画像の二倍の大きさの、空の画像を生成する。

 \param src 元画像
 \return 出力先画像
 */
Mat MakeOutputImage(const Mat& src)
  { return Mat::zeros(src.size().height, src.size().width * 2, src.type()); }
/*!
 \brief エラーメッセージを表示し、終了する

 \param error_message エラーメッセージ
 */
void ExitFailure(const char* const error_message) {
  cerr << "error: " << error_message << endl;
  exit(EXIT_FAILURE);
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
 \return カーネル
 */
Mat SetOperators(Mat kernel, ifstream& stream)
  { return SetOperators_impl(kernel, ::Rewind(stream)); }
/*!
 \brief カーネルの全ての要素へオペレータをセット。

 \param kernel カーネル
 \param stream ファイルストリーム
 \return カーネル
 */
Mat SetOperators_impl(Mat kernel, ifstream& stream) {
  string line;
  for (int i = 0;
       i < kernel.rows && getline(stream, line) && ::CheckKernelLine(line);
       ++i)
    { SetOperator(kernel.row(i), line, kernel.cols); }

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
  auto show_impl =
    bind(::HandleEmpty,
         bind(::MakeOutputImage, _1),
         "failed to create an output image") >=
    bind(::Combine, _1, src, filtered) >=
    ::Show;
  return show_impl(src);
};

auto test_filter = [](int argc, char** argv) {
  auto test_filter_impl =
    bind(::HandleEmpty, bind(get_image, _1, _2), "failed to read an image") >=
    [argc, argv](Mat src) {
      auto impl =
        bind(::HandleEmpty,
             bind(get_kernel, _1, _2),
             "failed to read a kernel") >=
        bind(::Filter, src, _1) >=
        bind(show, src, _1);
      return impl(argc, argv);
    };

  return test_filter_impl(argc, argv);
};
}  // namespace

int main(int argc, char** argv) {
  try {
    exit((argc >= 2)? ::GetExitCode(::test_filter(argc, argv)): ::Help());
  } catch(const exception& e) {
    ::ExitFailure(e.what());
  } catch(...) {
    ::ExitFailure("some exception was thrown");
  }

  return 0;
}

