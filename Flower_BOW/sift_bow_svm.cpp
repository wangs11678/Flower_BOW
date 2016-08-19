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

const string kVocabularyFile("vocabulary.xml.gz");
const string kBowImageDescriptorsDir("/bagOfWords");
const string kSvmsDirs("/svms");

int main(int argc, char* argv[])
{
	// read params
	int	wordCount(1000); //字典大小
	string databaseDir = "data\\train"; //训练集目录
	string testPicturePath = "data\\test"; //测试集目录
	string resultDir = "result"; //存放结果目录

	string detectorType("SIFT"); //检测子
	string descriptorType("SIFT"); //描述子
	string matcherType("FlannBased"); //匹配器

	cv::initModule_nonfree();

	string bowImageDescriptorsDir = resultDir + kBowImageDescriptorsDir; //bow特征存放目录(result/bagOfWords)
	string svmsDir = resultDir + kSvmsDirs; //svm分类器存放目录(result/svms)
	MakeDir(resultDir);
	MakeDir(bowImageDescriptorsDir);
	MakeDir(svmsDir);

	vector<string> categories;
	GetDirList(databaseDir, &categories); //求databaseDir下的目录(种类)
	
	Ptr<FeatureDetector> detector = FeatureDetector::create(descriptorType);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(descriptorType);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcherType);
	if (detector.empty() || extractor.empty() || matcher.empty())
	{
		cout << "feature detector or descriptor extractor or descriptor matcher cannot be created." << endl;
	}

	Mat vocabulary;
	string vocabularyFile = resultDir + '\\' + kVocabularyFile; //vocabulary存放目录(result/vocabulary.xml.gz)

	//生成字典
	vocabulary = BuildVocabulary(databaseDir, categories, vocabularyFile, detector, extractor, wordCount);

	Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(extractor, matcher);
	bowExtractor -> setVocabulary(vocabulary);

	//Samples这个map的key就是某个类别，value就是这个类别中所有图片的bag of words
	map<string, Mat> samples;//key: category name, value: histogram
	
	ComputeBowImageDescriptors(databaseDir, categories, detector, bowExtractor, bowImageDescriptorsDir,  &samples);
	
	vector<string> testCategories;
	GetDirList(testPicturePath, &testCategories);

	int sum = 0;
	int right = 0;
	//测试
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

			category = ClassifyBySvm(testPictureDescriptor, samples, svmsDir);

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


