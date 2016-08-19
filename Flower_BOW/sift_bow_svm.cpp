/*
* Flower Classification
* ������Bag of words
* ��������
* 1. ��ȡѵ������ͼƬ��feature��
* 2. ����Щfeature�۳�n�ࡣ��n���е�ÿһ����൱����ͼƬ�ġ����ʡ���
*    ���е�n����𹹳ɡ��ʻ�����ҵ�ʵ����nȡ1000�����ѵ�����ܴ�Ӧ����ȡֵ��
* 3. ��ѵ�����е�ͼƬ����bag of words�����ǽ�ͼƬ�е�feature�鵽��ͬ�����У�
*    Ȼ��ͳ��ÿһ���feature��Ƶ�ʡ����൱��ͳ��һ���ı���ÿһ�����ʳ��ֵ�Ƶ�ʡ�
* 4. ѵ��һ���������������ÿ��ͼƬ��bag of words��Ϊfeature vector��
*    ������ͼƬ�������Ϊlabel��
* 5. ����δ֪����ͼƬ����������bag of words��ʹ��ѵ���ķ��������з��ࡣ
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
	int	wordCount(1000); //�ֵ��С
	string databaseDir = "data\\train"; //ѵ����Ŀ¼
	string testPicturePath = "data\\test"; //���Լ�Ŀ¼
	string resultDir = "result"; //��Ž��Ŀ¼

	string detectorType("SIFT"); //�����
	string descriptorType("SIFT"); //������
	string matcherType("FlannBased"); //ƥ����

	cv::initModule_nonfree();

	string bowImageDescriptorsDir = resultDir + kBowImageDescriptorsDir; //bow�������Ŀ¼(result/bagOfWords)
	string svmsDir = resultDir + kSvmsDirs; //svm���������Ŀ¼(result/svms)
	MakeDir(resultDir);
	MakeDir(bowImageDescriptorsDir);
	MakeDir(svmsDir);

	vector<string> categories;
	GetDirList(databaseDir, &categories); //��databaseDir�µ�Ŀ¼(����)
	
	Ptr<FeatureDetector> detector = FeatureDetector::create(descriptorType);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(descriptorType);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcherType);
	if (detector.empty() || extractor.empty() || matcher.empty())
	{
		cout << "feature detector or descriptor extractor or descriptor matcher cannot be created." << endl;
	}

	Mat vocabulary;
	string vocabularyFile = resultDir + '\\' + kVocabularyFile; //vocabulary���Ŀ¼(result/vocabulary.xml.gz)

	//�����ֵ�
	vocabulary = BuildVocabulary(databaseDir, categories, vocabularyFile, detector, extractor, wordCount);

	Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(extractor, matcher);
	bowExtractor -> setVocabulary(vocabulary);

	//Samples���map��key����ĳ�����value����������������ͼƬ��bag of words
	map<string, Mat> samples;//key: category name, value: histogram
	
	ComputeBowImageDescriptors(databaseDir, categories, detector, bowExtractor, bowImageDescriptorsDir,  &samples);
	
	vector<string> testCategories;
	GetDirList(testPicturePath, &testCategories);

	int sum = 0;
	int right = 0;
	//����
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
			cvWaitKey(100); // ��ͣ0.1s��ʾͼ��
			
			cout<<endl;
		}	
	}
	cout<<"Total test image: "<<sum<<endl;
	cout<<"Correct prediction: "<<right<<endl;	
	cout<<"Accuracy: "<<(double(right)/sum)<<endl;
	return 0;
}


