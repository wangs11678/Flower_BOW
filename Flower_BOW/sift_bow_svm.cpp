/*
* Flower Classification
* 方法：Bag of words
* 步骤描述
* 1. 提取训练集中图片的feature。
* 2. 将这些feature聚成n类。这n类中的每一类就相当于是图片的“单词”，
*    所有的n个类别构成“词汇表”。我的实现中n取1000，如果训练集很大，应增大取值。
* 3. 对训练集中的图片构造bag of words，就是将图片中的feature归到不同的类中，
*    然后统计每一类的feature的频率。这相当于统计一个文本中每一个单词出现的频率。
* 4. 训练一个多类分类器，将每张图片的bag of words作为feature vector，
*    将该张图片的类别作为label。
* 5. 对于未知类别的图片，计算它的bag of words，使用训练的分类器进行分类。
*/

// windows api
#include <Windows.h>
#include <tchar.h>
#include <strsafe.h>
#pragma comment( lib, "User32.lib")
// c api
#include <stdio.h>
#include <string.h>
#include <assert.h>
// c++ api
#include <string>
#include <map>
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
// opencv api
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;

// some utility functions
void MakeDir(const string& filepath);
void help(const char* progName);
void GetDirList(const string& directory, vector<string>* dirlist);
void GetFileList(const string& directory, vector<string>* filelist);

const string kVocabularyFile("vocabulary.xml.gz");
const string kBowImageDescriptorsDir("/bagOfWords");
const string kSvmsDirs("/svms");

class Params
{
public:
	Params(): wordCount(1000), 
			  detectorType("SIFT"),
			  descriptorType("SIFT"), 
			  matcherType("FlannBased"){ }

	int		wordCount;
	string	detectorType;
	string	descriptorType;
	string	matcherType;
};

/*
 * loop through every directory 
 * compute each image's keypoints and descriptors
 * train a vocabulary
 */
Mat BuildVocabulary(const string& databaseDir, 
					const vector<string>& categories, 
					const Ptr<FeatureDetector>& detector, 
					const Ptr<DescriptorExtractor>& extractor,
					int wordCount)
{
	Mat allDescriptors;
	for (int index = 0; index != categories.size(); ++index)
	{
		cout << "processing category " << categories[index] << endl;
		string currentCategory = databaseDir + '\\' + categories[index];
		vector<string> filelist;
		GetFileList(currentCategory, &filelist);
		for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex += 10)
		{			
			string filepath = currentCategory + '\\' + *fileindex;
			Mat image = imread(filepath);
			if (image.empty())
			{
				continue; // maybe not an image file
			}
			vector<KeyPoint> keyPoints;
			Mat descriptors;
			detector -> detect(image, keyPoints);
			extractor -> compute(image, keyPoints, descriptors);
			if (allDescriptors.empty())
			{
				allDescriptors.create(0, descriptors.cols, descriptors.type());
			}
			allDescriptors.push_back(descriptors);
		}
		cout << "done processing category " << categories[index] << endl;
	}
	assert(!allDescriptors.empty());
	cout << "build vocabulary..." << endl;
	BOWKMeansTrainer bowTrainer(wordCount);
	Mat vocabulary = bowTrainer.cluster(allDescriptors);
	cout << "done build vocabulary..." << endl;
	return vocabulary;
}

// bag of words of an image as its descriptor, not keypoint descriptors
void ComputeBowImageDescriptors(const string& databaseDir,
								const vector<string>& categories, 
								const Ptr<FeatureDetector>& detector,
								Ptr<BOWImgDescriptorExtractor>& bowExtractor,
								const string& imageDescriptorsDir,
								map<string, Mat>* samples)
{	
	for (auto i = 0; i != categories.size(); ++i)
	{
		string currentCategory = databaseDir + '\\' + categories[i];
		vector<string> filelist;
		GetFileList(currentCategory, &filelist);	
		for (auto fileitr = filelist.begin(); fileitr != filelist.end(); ++fileitr)
		{
			string descriptorFileName = imageDescriptorsDir + "\\" + categories[i] + "\\" + (*fileitr) + ".xml.gz";
			MakeDir(imageDescriptorsDir + "/" + categories[i]);
			FileStorage fs(descriptorFileName, FileStorage::READ);
			Mat imageDescriptor;
			if (fs.isOpened())
			{ 
				// already cached
				fs["imageDescriptor"] >> imageDescriptor;
			} 
			else
			{
				cout<<"Computing the bag of words of class "<<categories[i]<<endl;

				string filepath = currentCategory + '\\' + *fileitr;
				Mat image = imread(filepath);
				if (image.empty())
				{
					continue; // maybe not an image file
				}
				vector<KeyPoint> keyPoints;
				detector -> detect(image, keyPoints);
				bowExtractor -> compute(image, keyPoints, imageDescriptor);
				fs.open(descriptorFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "imageDescriptor" << imageDescriptor;
				}
			}
			//判断samples的string中有无categories[i].（samples是map<string, Mat>*）
			if (samples -> count(categories[i]) == 0)
			{
				(*samples)[categories[i]].create(0, imageDescriptor.cols, imageDescriptor.type());
			}
			(*samples)[categories[i]].push_back(imageDescriptor);
		}
	}
}

void TrainSvm(const map<string, Mat>& samples, const string& category, const CvSVMParams& svmParams, CvSVM* svm)
{
	Mat allSamples(0, samples.at(category).cols, samples.at(category).type());
	Mat responses(0, 1, CV_32SC1);
	//assert(responses.type() == CV_32SC1);
	allSamples.push_back(samples.at(category));
	Mat posResponses(samples.at(category).rows, 1, CV_32SC1, Scalar::all(1)); 
	responses.push_back(posResponses);
	
	for (auto itr = samples.begin(); itr != samples.end(); ++itr)
	{
		if (itr -> first == category)
		{
			continue;
		}
		allSamples.push_back(itr -> second);
		Mat response(itr -> second.rows, 1, CV_32SC1, Scalar::all( -1 ));
		responses.push_back(response);		
	}
	svm -> train(allSamples, responses, Mat(), Mat(), svmParams);
}

// using 1-vs-all method, train an svm for each category.
// choose the category with the biggest confidence
string ClassifyBySvm(const Mat& queryDescriptor, const map<string, Mat>& samples, const string& svmDir)
{
	string category;
	SVMParams svmParams;
	int sign = 0; //sign of the positive class
	float confidence = -FLT_MAX;
	for (auto itr = samples.begin(); itr != samples.end(); ++itr)
	{
		CvSVM svm;
		string svmFileName = svmDir + "\\" + itr -> first + ".xml.gz";
		FileStorage fs(svmFileName, FileStorage::READ);
		if (fs.isOpened())
		{ 
			// exist a previously trained svm
			fs.release();
			svm.load(svmFileName.c_str());
		} 
		else
		{
			TrainSvm(samples, itr->first, svmParams, &svm);
			if (!svmDir.empty())
			{
				svm.save(svmFileName.c_str());
			}
		}
		// determine the sign of the positive class
		if (sign == 0)
		{
			float scoreValue = svm.predict(queryDescriptor, true);
			float classValue = svm.predict(queryDescriptor, false);
			sign = (scoreValue < 0.0f) == (classValue < 0.0f)? 1 : -1;
		}
		float curConfidence = sign * svm.predict(queryDescriptor, true);
		if (curConfidence > confidence)
		{
			confidence = curConfidence;
			category = itr -> first;
		}
	}
	return category;
}

string ClassifyByMatch(const Mat& queryDescriptor, const map<string, Mat>& samples)
{
	// find the best match and return category of that match
	int normType = NORM_L2;
	Ptr<DescriptorMatcher> histogramMatcher = new BFMatcher(normType);
	float distance = FLT_MAX;
	struct Match
	{
		string category;
		float distance;
		Match(string c, float d): category(c), distance(d){}
		bool operator<(const Match& rhs) const
		{ 
			return distance > rhs.distance; 
		}
	};
	priority_queue<Match, vector<Match>> matchesMinQueue;
	const int numNearestMatch = 9;
	for (auto itr = samples.begin(); itr != samples.end(); ++itr)
	{
		vector<vector<DMatch>> matches;
		histogramMatcher -> knnMatch(queryDescriptor, itr ->second, matches, numNearestMatch);
		for (auto itr2 = matches[0].begin(); itr2 != matches[0].end(); ++ itr2)
		{
			matchesMinQueue.push(Match( itr -> first, itr2 -> distance));
		}
	}
	string category;
	int maxCount = 0;
	map<string, size_t> categoryCounts;
	size_t select = std::min(static_cast<size_t>(numNearestMatch), matchesMinQueue.size());
	for (size_t i = 0; i < select; ++i)
	{
		string& c = matchesMinQueue.top().category;
		++categoryCounts[c];
		int currentCount = categoryCounts[c];
		if (currentCount > maxCount)
		{
			maxCount = currentCount;
			category = c;
		}
		matchesMinQueue.pop();
	}
	return category;
}

int main(int argc, char* argv[])
{
	if (argc != 5 && argc != 8)
	{
		help(argv[0]);
		return -1;
	}
	// read params
	Params params;
	string method = argv[1];
	string databaseDir = argv[2];
	string testPicturePath = argv[3];
	string resultDir = argv[4];
	if (argc == 8)
	{
		params.detectorType = argv[5];
		params.descriptorType = argv[6];
		params.matcherType = argv[7];
	}

	//string method = "svm";
	//string databaseDir = "data\\train";
	//string testPicturePath = "data\\test";
	//string resultDir = "result";

	cv::initModule_nonfree();

	string bowImageDescriptorsDir = resultDir + kBowImageDescriptorsDir;
	string svmsDir = resultDir + kSvmsDirs;
	MakeDir(resultDir);
	MakeDir(bowImageDescriptorsDir);
	MakeDir(svmsDir);

	// key: image category name
	// value: histogram of image
	vector<string> categories;
	GetDirList(databaseDir, &categories);
	
	Ptr<FeatureDetector> detector = FeatureDetector::create(params.descriptorType);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(params.descriptorType);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(params.matcherType);
	if (detector.empty() || extractor.empty() || matcher.empty())
	{
		cout << "feature detector or descriptor extractor or descriptor matcher cannot be created.\nMaybe try other types?" << endl;
	}
	Mat vocabulary;
	string vocabularyFile = resultDir + '\\' + kVocabularyFile;

	//OpenCV FileStorage类读(写)XML/YML文件
	FileStorage fs(vocabularyFile, FileStorage::READ);

	if (fs.isOpened())
	{
		//将保存在fs对象指定yml文件下的vocabulary标签下的数据读到vocabulary矩阵
		fs["vocabulary"] >> vocabulary;
	} 
	else
	{
		vocabulary = BuildVocabulary(databaseDir, categories, detector, extractor, params.wordCount);
		//OpenCV FileStorage类(读)写XML/YML文件
		FileStorage fs(vocabularyFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			//将vocabulary矩阵保存在fs对象指定的yml文件的vocabulary标签下
			fs << "vocabulary" << vocabulary;
		}
	}
	Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(extractor, matcher);
	bowExtractor -> setVocabulary(vocabulary);

	//Samples这个map的key就是某个类别，value就是这个类别中所有图片的bag of words
	map<string, Mat> samples;//key: category name, value: histogram
	
	ComputeBowImageDescriptors(databaseDir, categories, detector, bowExtractor, bowImageDescriptorsDir,  &samples);
	
	vector<string> testCategories;
	GetDirList(testPicturePath, &testCategories);

	int sum = 0;
	int right = 0;
	for (auto i = 0; i != testCategories.size(); ++i)
	{		
		string currentCategory = testPicturePath + '\\' + testCategories[i];
		vector<string> filelist;
		GetFileList(currentCategory, &filelist);
		for (auto fileindex = filelist.begin(); fileindex != filelist.end(); ++fileindex)
		{			
			string filepath = currentCategory + '\\' + *fileindex;
			Mat image = imread(filepath);
			
			cout << "Classify image " << *fileindex << "." << endl;

			vector<KeyPoint> keyPoints;
			detector -> detect(image, keyPoints);
			Mat testPictureDescriptor;
			bowExtractor -> compute(image, keyPoints, testPictureDescriptor);
			string category;
			if (method == "svm")
			{
				category = ClassifyBySvm(testPictureDescriptor, samples, svmsDir);
			}
			else 
			{
				category = ClassifyByMatch(testPictureDescriptor, samples);
			}

			if(category == testCategories[i])
			{
				right++;
			}
			sum++;
			cout << "I think it should be " << category << "." << endl;

			//显示图像
			//CvFont font;
			//cvInitFont(&font,CV_FONT_VECTOR0,1,1,0,1,8);
			//在图像中显示文本字符串
			//cvPutText(image,"HELLO",cvPoint(20,20),&font,CV_RGB(255,255,255));

			destroyAllWindows();
			string info = "pred: " + category;					
			imshow(info, image);
			cvWaitKey(100); // 暂停0.1s显示图像
			
			cout<<endl;
		}	
	}
	cout<<"Total test image: "<<sum<<endl;
	cout<<"Correct prediction: "<<right<<endl;	
	cout<<"Accuracy: "<<(double(right)/sum)<<endl;
	return 0;
}

void help(const char* progName)
{
	cout << "Usage: \n"
		 << progName << " {classify method} {query image} {image set path} {result directory}\n"
		 << "or: \n"
		 << progName << " {classify method} {image set path} {result directory} {feature detector} {descriptor extrator} {descriptor matcher}\n"
		 << "\n"
		 << "Input parameters: \n"
		 << "{classify method}			\n	Method used to classify image, can be one of svm or match.\n"
		 << "{query image}				\n	Path to query image.\n"
		 << "{image set path}			\n	Path to image training set, organized into categories, like Caltech 101.\n"
		 << "{result directory}			\n	Path to result directory.\n"
		 << "{feature detector}			\n	Feature detector name, should be one of\n"
		 <<	"	FAST, STAR, SIFT, SURF, MSER, GFTT, HARRIS.\n"
		 << "{descriptor extractor}		\n	Descriptor extractor name, should be one of \n"
		 <<	"	SURF, OpponentSIFT, SIFT, OpponentSURF, BRIEF.\n"
		 << "{descriptor matcher}		\n	Descriptor matcher name, should be one of\n"
		 << "	BruteForce, BruteForce-L1, FlannBased, BruteForce-Hamming, BruteForce-HammingLUT.\n";
}

void MakeDir(const string& filepath)
{
	TCHAR path[MAX_PATH];
#ifdef _UNICODE
	MultiByteToWideChar(CP_ACP, NULL, filepath.c_str(), -1, path, MAX_PATH);
#else
	StringCchCopy(path, MAX_PATH, filepath.c_str());
#endif
	CreateDirectory(path, 0);
}

void ListDir(const string& directory, bool (*filter)(const WIN32_FIND_DATA& entry), vector<string>* entries)
{
	WIN32_FIND_DATA entry;
	TCHAR dir[MAX_PATH];
	HANDLE hFind = INVALID_HANDLE_VALUE;
#ifdef _UNICODE
	MultiByteToWideChar(CP_ACP, NULL, directory.c_str(), -1, dir, MAX_PATH);
	char dirName[MAX_PATH];
#endif
	StringCchCat(dir, MAX_PATH, _T("\\*"));
	hFind = FindFirstFile(dir, &entry);
	do
	{
		if ( filter(entry))
		{
		#ifdef _UNICODE
			WideCharToMultiByte(CP_ACP, NULL, entry.cFileName, -1, dirName, MAX_PATH, NULL, NULL);
			entries -> push_back(dirName);
		#else
			entries -> push_back(entry.cFileName);
		#endif
		}
	} while (FindNextFile(hFind, &entry) != 0);
}

void GetDirList(const string& directory, vector<string>* dirlist)
{
	ListDir(directory, [](const WIN32_FIND_DATA& entry)
	{ 
		return (entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && 
				lstrcmp(entry.cFileName, _T(".")) != 0 &&
				lstrcmp(entry.cFileName, _T("..")) != 0;
	}, dirlist);
}

void GetFileList(const string& directory, vector<string>* filelist)
{
	ListDir(directory, [](const WIN32_FIND_DATA& entry)
	{ 
		return !(entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
	}, filelist);
}