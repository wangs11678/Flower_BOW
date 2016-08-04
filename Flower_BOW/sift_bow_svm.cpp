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

#include "utils.h"
#include "vocabulary.h"
#include "bow.h"
#include "svm.h"
#include "match.h"

const string kVocabularyFile("vocabulary.xml.gz");
const string kBowImageDescriptorsDir("/bagOfWords");
const string kSvmsDirs("/svms");

int main(int argc, char* argv[])
{
	// read params
	int	wordCount(1000);
	string method = "svm";
	string databaseDir = "data\\train";
	string testPicturePath = "data\\test";
	string resultDir = "result";	
	string detectorType("SIFT");
	string descriptorType("SIFT");
	string matcherType("FlannBased");

	cv::initModule_nonfree();

	string bowImageDescriptorsDir = resultDir + kBowImageDescriptorsDir;
	string svmsDir = resultDir + kSvmsDirs;
	MakeDir(resultDir);
	MakeDir(bowImageDescriptorsDir);
	MakeDir(svmsDir);

	vector<string> categories;
	GetDirList(databaseDir, &categories);
	
	Ptr<FeatureDetector> detector = FeatureDetector::create(descriptorType);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(descriptorType);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcherType);
	if (detector.empty() || extractor.empty() || matcher.empty())
	{
		cout << "feature detector or descriptor extractor or descriptor matcher cannot be created." << endl;
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
		vocabulary = BuildVocabulary(databaseDir, categories, detector, extractor, wordCount);
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
			cout << "predicted value: " << category << "." << endl;

			destroyAllWindows();
			string info = "predicted: " + category;					
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


