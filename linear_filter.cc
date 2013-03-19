#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::atof;
using std::count;
using std::cout;
using std::cerr;
using std::endl;
using std::getline;
using std::string;
using std::stringstream;
using std::ifstream;
using cv::filter2D;
using cv::imread;
using cv::Mat;
using cv::namedWindow;
using cv::Rect;
using cv::waitKey;

namespace {
/*
 Set operators to a kernel row from a string
 \param row a kernel row.
 \param line a string
 \param size a kernel size
*/
void setOperator(Mat row, const string& line, uint64_t size) {
  stringstream line_stream(line);
  for(uint64_t i = 0; i < size; ++i) {
    if(line_stream.good() && !line_stream.eof()) {
      string op;
      getline(line_stream, op, ',');
      row.at<double>(i) =  static_cast<double>(atof(op.c_str()));
    } else { break; }
  }
};
/*!
 Get a kernel matrix from a csv file.
 \param filename csv file name
 \return a kernel matrix
*/
cv::Mat getKernel(const std::string& filename) {
  ifstream stream;
  stream.open(filename);

  if(stream.good()) {
    string line;
    getline(stream, line);

    uint64_t size = count(line.begin(), line.end(), ',') + 1;

    Mat kernel(size, size, cv::DataType<double>::type);
    if(kernel.data == NULL) { return Mat::zeros(1, 1, CV_8S); }

    setOperator(kernel.row(0), line, size);

    for(uint64_t i = 1; i < size; ++i) {
      if(stream.good() && !stream.eof()) {
        getline(stream, line);
        setOperator(kernel.row(i), line, size);
      } else { break; }
    }

    return kernel;
  } else { return Mat::zeros(1, 1, CV_8S); }
}
/*!
 Get a filtered matrix.
 \param src an original matrix.
 \param kernel a kernel.
 \return a fitered matrix.
*/
cv::Mat filter(const cv::Mat& src, const cv::Mat& kernel) {
  Mat filtered;
  src.copyTo(filtered);

  filter2D(src, filtered, src.depth(), kernel, cv::Point(0, 0));
  return filtered;
}
/*!
 Show an original image and a filtered image on GUI.
 \param original an original image matrix.
 \param filtered a filtered image matrix.
 \return always EXIT_SUCCESS
*/
int showImage(const cv::Mat& original, const cv::Mat& filtered) {
  Mat output = Mat::zeros(
    original.size().height, original.size().width * 2, original.type()
  );

  original.copyTo(Mat(output, Rect(0, 0, original.cols, original.rows)));
  filtered.copyTo(
    Mat(output, Rect(original.cols, 0, original.cols, original.rows))
  );

  string window_name("linear_filter");
  namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  imshow(window_name, output);

  waitKey(0);

  return EXIT_SUCCESS;
}
/*!
 Show an empty window to show error messages on title.
 When an empty window was closed, show error messages on an console.
 \param window_name error messages
 \return always EXIT_FAILURE
*/
int showErrorWindow(const std::string& window_name) {
  namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  waitKey(0);

  cerr << window_name << endl;

  return EXIT_FAILURE;
}
} // namespace

int main(int argc, char** argv) { 
  if(argc == 3) {
    Mat original = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    exit(
      (original.data != NULL)?
        ::showImage(original, ::filter(original, ::getKernel(string(argv[2])))):
        ::showErrorWindow(string("Could not open or find the image"))
    );
  } else {
    exit(
      ::showErrorWindow(string("Usage: linear_filter image_file filter_csv"))
   );
  }

  return 0;
}

