// opencv api
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

void TrainSvm(const map<string, Mat>& samples, const string& category, const CvSVMParams& svmParams, CvSVM* svm);
string ClassifyBySvm(const Mat& queryDescriptor, const map<string, Mat>& samples, const string& svmDir);