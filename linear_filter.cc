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
 Set operators to a kernel row from a string.
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
      // Set an operator parameter.
      row.at<double>(i) =  static_cast<double>(atof(op.c_str()));
    } else { break; }
  }
};
/*!
 Get a kernel matrix from a csv file.
 \param filename csv file name
 \return a kernel matrix. If had some error, return empty matrix.
*/
cv::Mat getKernel(const std::string& filename) {
  ifstream stream;
  stream.open(filename);

  if(stream.good()) {
    string line;
    getline(stream, line);

    // Get filter size.
    uint64_t size = count(line.begin(), line.end(), ',') + 1;
    // Allocate a kernel matrix.
    Mat kernel = Mat::zeros(size, size, cv::DataType<double>::type);

    if(kernel.data == NULL) {
      return Mat();
    } else {
      setOperator(kernel.row(0), line, size);

      for(uint64_t i = 1; i < size; ++i) {
        if(stream.good() && !stream.eof()) {
          getline(stream, line);
          setOperator(kernel.row(i), line, size);
        } else { break; }
      }

      return kernel;
    }
  } else { return Mat(); }
}
/*!
 Get a filtered matrix.
 \param src an original matrix.
 \param kernel a kernel.
 \return a fitered matrix. If an input matrix was empty, return an empty matrix.
*/
cv::Mat filter(const cv::Mat& src, const cv::Mat& kernel) {
  if(src.data != NULL && kernel.data != NULL) {
    Mat filtered;
    src.copyTo(filtered);

    filter2D(src, filtered, src.depth(), kernel, cv::Point(0, 0));
    return filtered;
  } else { return Mat(); }
}
/*!
 Show an empty window to show error messages on title.
 When an empty window was closed, show error messages on an console.
 \param window_name error messages
 \return always EXIT_FAILURE
*/
int showErrorWindow(const std::string& error_message) {
  namedWindow(error_message, CV_WINDOW_AUTOSIZE);
  waitKey(0);

  cerr << error_message << endl;

  return EXIT_FAILURE;
}
/*!
 Show an original image and a filtered image on GUI.
 \param original an original image matrix.
 \param filtered a filtered image matrix.
 \return always EXIT_SUCCESS
*/
int showImageWindow(const cv::Mat& original, const cv::Mat& filtered) {
  Mat output = Mat::zeros(
    original.size().height, original.size().width * 2, original.type()
  );

  // Combine an original image and a filtered image.
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
int showWindow(const cv::Mat& original, const cv::Mat& filtered) {
  if(original.data == NULL) {
    return showErrorWindow(string("failed to open the image."));
  } else if(filtered.data == NULL) {
    return showErrorWindow(string("failed to filter the image."));
  } else { return showImageWindow(original, filtered); }
}
//int showImageWindow(
/*
 Get image file name from program options.
 \param agrc argc
 \param argv argv
 \return image file name
*/
std::string getImageFilename(int argc, char** argv)
  { return (argc == 2)? string("input.jpg"): string(argv[1]); }
/*!
 Get kernel file name from program options.
 \param argc argc
 \param argv argv
 \return kernel file name
 */
std::string getKernelFilename(int argc, char** argv)
  { return (argc == 2)? string(argv[1]): string(argv[2]); }
} // namespace

int main(int argc, char** argv) { 
  if(argc == 2 || argc == 3) {
    Mat original = imread(
      ::getImageFilename(argc, argv), CV_LOAD_IMAGE_GRAYSCALE
    );

    exit(
      ::showWindow(
        original,
        ::filter(original, ::getKernel(::getKernelFilename(argc, argv)))
      )
    );
  } else {
    exit(
      ::showErrorWindow(string("Usage: linear_filter image_file filter_csv"))
   );
  }

  return 0;
}

