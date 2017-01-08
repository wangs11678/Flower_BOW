/*
* Flower Classification
* ������Bag of Words
* ��������
* 1. ��ȡѵ������ͼƬ��feature��
* 2. ����Щfeature�۳�n�ࡣ��n���е�ÿһ����൱����ͼƬ�ġ����ʡ���
*    ���е�n����𹹳ɡ��ʻ����������nȡ1000�����ѵ�����ܴ�Ӧ����ȡֵ��
* 3. ��ѵ�����е�ͼƬ����Bag of Words�����ǽ�ͼƬ�е�feature�鵽��ͬ�����У�
*    Ȼ��ͳ��ÿһ���feature��Ƶ�ʡ����൱��ͳ��һ���ı���ÿһ�����ʳ��ֵ�Ƶ�ʡ�
* 4. ѵ��һ���������������ÿ��ͼƬ��Bag of Words��Ϊfeature vector��
*    ������ͼƬ�������Ϊlabel��
* 5. ����δ֪����ͼƬ����������Bag of Words��ʹ��ѵ���ķ��������з��ࡣ
*/

// c api
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
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

//socket api
#include <sys/types.h>          /* See NOTES */
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <strings.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <sys/select.h>

//thread api
#include <pthread.h>

using namespace std;
using namespace cv;

void getdirlist(char *path, vector<string> &dirlist);
void getfilelist(char *path, vector<string> &filelist);

const string kVocabularyFile("vocabulary.xml.gz");
const string kBowImageDescriptorsDir("/bagOfWords");
const string kSvmsDirs("/svms");

float result = 0; //the result of the svm, 1:red 2:white 3:bud 

/****************************socket**************************************/
typedef struct DataPackage
{
	int packageLen;
	int commandNo;
	int deviceNo;
	int messageID;
	float message;
	char* pend;
}DataPackage;

typedef struct PicPackage
{
	int packageLen;
	int commandNo;
	int deviceNo;
	int messageID;
    //char * picture;
	char picture[1000000];
	char* pend;
}PicPackage;


int cmdNo1 = 101; //�����ţ��������ݣ�
int cmdNo2 = 102; //�����ţ�����ͼƬ��
int devNo = 0; //�豸��
int messageID = 0; //��ϢID
float message = 0; //����result����
char picturePath[40]; //����ͼƬpath
char pend[10] = "pend"; //��������


void sendPicture(int sockClient,int cmdNo, int devNo, int msgID, float message, char* picPath, char* end)
{
/********************************send result*****************************************/	
    char buff[100] = {0};

    if(cmdNo==101)
    {
		cout<<"..........Start send result data..........."<<endl;	
        DataPackage *dp = (DataPackage*)malloc(sizeof(DataPackage));        
		dp->packageLen = 24;
		dp->commandNo = cmdNo;
		dp->deviceNo = devNo;
		dp->messageID = msgID;
		dp->message = message;
		dp->pend = end;

		int n;
       	//��������  
        n = send(sockClient, (char*)dp, sizeof(struct DataPackage), 0); 
 		if (n == 0){
        	printf("send() failed!\n");
       	}
		//���շ�������Ӧ  
		cout<<"Server Response: ";      
        recv(sockClient, buff, sizeof(buff), 0);  
        cout<<buff<<endl;  
		cout<<"Send data finished."<<endl;
		free(dp);		
    }
/********************************send picture*****************************************/
	else if(cmdNo==102)
	{
		//����Ϣ��ʽȫ���������
		cout<<"..........Start send the picture........"<<endl;
        FILE *fp;
		fp=fopen(picPath, "rb+");
		fseek(fp, 0, SEEK_END);
		int fend=ftell(fp);
		//cout<<fend<<endl;
	
		fseek(fp, 0, 0);		
        PicPackage *pp = (PicPackage*)malloc(sizeof(PicPackage));
		int ret;
		pp->packageLen = fend+20;
		//pp.packageLen = 24;
		pp->commandNo = cmdNo; //������
		pp->deviceNo = devNo; //�豸���
		pp->messageID = msgID; //��ϢID
		//pp.picture = &msg[0]; //ͼƬ����
		pp->pend = end; //��������			
		bzero(pp->picture, sizeof(pp->picture));              
        fread(pp->picture, fend, 1, fp);		          				               
        ret=send(sockClient, (char*)pp, pp->packageLen, 0); //�������͸�������               
        //printf("%d\n",ret);
        if (ret == 0){
        	printf("send() failed!\n");
       	}
		//���շ�������Ӧ  
		cout<<"Server Response: ";
        recv(sockClient, buff, sizeof(buff), 0);  
        cout<<buff<<endl;  
		cout<<"Send picture finished."<<endl; 
		fclose(fp);
		free(pp);              			
	}
}

/*********************************client thread*****************************/    
void * thread_result_client(void *)
{
	int client_fd;
	if(-1 == (client_fd=socket(AF_INET, SOCK_STREAM, 0))){
		perror("create socket error!\n");
		exit(-1);
	}
	struct sockaddr_in client_socket = {0};
	client_socket.sin_family = AF_INET;
	client_socket.sin_port = htons(atoi("20101"));
	char ip[20] = "192.168.1.210";
	// char ip[20]="202.115.13.252";
	if(0 == inet_aton(ip, &client_socket.sin_addr)){
		printf("inet_aton error!\n");
		exit(-1);
	}
	
	int len = sizeof(struct sockaddr);
	if(-1 == connect(client_fd,(struct sockaddr *) &client_socket, len)){
		perror("connect error!\n");
		exit(-1);
	}
		
	//���ͽ������orͼƬ 
    sendPicture(client_fd, cmdNo1, devNo, messageID, message, picturePath, pend);
	sendPicture(client_fd, cmdNo2, devNo, messageID, message, picturePath, pend);
		
	//�ر��׽���
	close(client_fd);
}


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
		string currentCategory = databaseDir + '/' + categories[index];
       	vector<string> filelist;
        getfilelist((char *)currentCategory.c_str(), filelist);
		
		cout<<(char *)currentCategory.c_str()<<endl;

    
	   	for (int j = 0; j != filelist.size(); j += 10)
		{			
			string filepath = currentCategory + '/' + filelist[j];
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
	for (int i = 0; i != categories.size(); ++i)
	{
		string currentCategory = databaseDir + '/' + categories[i];
		vector<string> filelist;

		getfilelist((char *)currentCategory.c_str(), filelist);
		
	    for (int j =0; j != filelist.size(); ++j)
		{
			string descriptorFileName = imageDescriptorsDir + '/' + categories[i] + '/' + filelist[j] + ".xml.gz";
			//MakeDir(imageDescriptorsDir + "/" + categories[i]);
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

				string filepath = currentCategory + '/' + filelist[j];
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
			//�ж�samples��string������categories[i].��samples��map<string, Mat>*��
			if (samples -> count(categories[i]) == 0)
			{
				(*samples)[categories[i]].create(0, imageDescriptor.cols, imageDescriptor.type());
			}
			(*samples)[categories[i]].push_back(imageDescriptor);
		}
	}
}

void TrainSvm(map<string, Mat>& samples, const string& category, const CvSVMParams& svmParams, CvSVM* svm)
{
	Mat allSamples(0, samples.at(category).cols, samples.at(category).type());
	Mat responses(0, 1, CV_32SC1);
	//assert(responses.type() == CV_32SC1);
	allSamples.push_back(samples.at(category));
	Mat posResponses(samples.at(category).rows, 1, CV_32SC1, Scalar::all(1)); 
	responses.push_back(posResponses);
	
	for (map<string, Mat>::iterator itr = samples.begin(); itr != samples.end(); ++itr)
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
string ClassifyBySvm(const Mat& queryDescriptor, map<string, Mat>& samples, const string& svmDir)
{
	string category;
	SVMParams svmParams;
	int sign = 0; //sign of the positive class
	float confidence = -FLT_MAX;
	for (map<string, Mat>::iterator itr = samples.begin(); itr != samples.end(); ++itr)
	{
		CvSVM svm;
		string svmFileName = svmDir + '/' + itr -> first + ".xml.gz";
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


//////////////////////   ClassifyByMatc   ///////////////////////////////////////
/*
string ClassifyByMatch(const Mat& queryDescriptor, map<string, Mat>& samples)
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
	priority_queue<Match, vector<Match> > matchesMinQueue;
	//priority_queue<Match> matchesMinQueue;
        const int numNearestMatch = 9;
	for (map<string, Mat>::iterator itr = samples.begin(); itr != samples.end(); ++itr)
	{
		
		
		vector<vector<DMatch> > matches;
		histogramMatcher -> knnMatch(queryDescriptor, itr ->second, matches, numNearestMatch);
		//for (auto itr2 = matches[0].begin(); itr2 != matches[0].end(); ++ itr2)
		//{
		//	matchesMinQueue.push(Match( itr -> first, itr2 -> distance));
		//}

                for (int j=0;j!= matches[0].size(); ++j)
		{
			matchesMinQueue.push(Match( itr -> first, matches[0][0].distance));
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
//////////////////////   ClassifyByMatc   ///////////////////////////////////////
*/

int main(int argc, char* argv[])
{
	Params params;
	//string method="svm";
	string databaseDir = "./data/train";
	string testPicturePath = "./data/test";
	string resultDir = "./result";
	
	cv::initModule_nonfree();

	string bowImageDescriptorsDir = resultDir + kBowImageDescriptorsDir;
	string svmsDir = resultDir + kSvmsDirs;
		
	vector<string> categories;
	
	getdirlist((char *)databaseDir.c_str(),categories);

	Ptr<FeatureDetector> detector = FeatureDetector::create(params.descriptorType);
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create(params.descriptorType);
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(params.matcherType);
	if (detector.empty() || extractor.empty() || matcher.empty())
	{
		cout << "feature detector or descriptor extractor or descriptor matcher cannot be created.\nMaybe try other types?" << endl;
	}
	
	Mat vocabulary;
	string vocabularyFile = resultDir + '/' + kVocabularyFile;

	//OpenCV FileStorage���(д)XML/YML�ļ�
	FileStorage fs(vocabularyFile, FileStorage::READ);

	if (fs.isOpened())
	{
		//��������fs����ָ��yml�ļ��µ�vocabulary��ǩ�µ����ݶ���vocabulary����
		fs["vocabulary"] >> vocabulary;
	} 
	else
	{
		vocabulary = BuildVocabulary(databaseDir, categories, detector, extractor, params.wordCount);
		//OpenCV FileStorage��(��)дXML/YML�ļ�
		FileStorage fs(vocabularyFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			//��vocabulary���󱣴���fs����ָ����yml�ļ���vocabulary��ǩ��
			fs << "vocabulary" << vocabulary;
		}
	}
	Ptr<BOWImgDescriptorExtractor> bowExtractor = new BOWImgDescriptorExtractor(extractor, matcher);
	bowExtractor -> setVocabulary(vocabulary);

	//Samples���map��key����ĳ�����value����������������ͼƬ��bag of words
	map<string, Mat> samples;//key: category name, value: histogram
	
	ComputeBowImageDescriptors(databaseDir, categories, detector, bowExtractor, bowImageDescriptorsDir,  &samples);
	
	vector<string> testCategories;
		
    getdirlist((char *)testPicturePath.c_str(), testCategories);
	int sum = 0;
	int right = 0;
	for (int i = 0; i != testCategories.size(); ++i)
	{		
		string currentCategory = testPicturePath + '/' + testCategories[i];
		vector<string> filelist;

        getfilelist((char *)currentCategory.c_str(), filelist);

	    for (int j=0; j != filelist.size(); ++j)
		{			
			string filepath = currentCategory + '/' + filelist[j];
                        
			Mat image = imread(filepath);
			cout << "Classify image " << filelist[j] << "." << endl;
			
			vector<KeyPoint> keyPoints;
			detector -> detect(image, keyPoints);
			Mat testPictureDescriptor;
			bowExtractor -> compute(image, keyPoints, testPictureDescriptor);
			string category;
		    //if (method == "svm")
		    //{
				category = ClassifyBySvm(testPictureDescriptor, samples, svmsDir);
		    //}
		    //else 
		    //{
			//	category = ClassifyByMatch(testPictureDescriptor, samples);
		    //}

			if(category == testCategories[i])
			{
				right++;
			}
			sum++;
			cout << "pred: " << category << "." << endl;

            //the result of the svm, 1:red 2:white 3:bud
                        
			if( category=="redFlower"){
            	result=1;
            }
			else if(category=="whiteFlower"){
                result=2;
            }
			else if(category=="budFlower"){
                result=3;
            }    
			//cout<<"the path is "<<filepath<<endl;      		                  
            //cout<<"the result is "<<result<<endl; 
			//cout<<i<<endl; 
            /////////////////////set the package value////////////////////////////////////			
			devNo = 1; //�豸��
			messageID = messageID+1; //��ϢID
			//cout<<messageID<<endl;
			message = result;
			//result = 0; //����result����
			//char picturePath[40]; //����ͼƬpath 
            strcpy(picturePath,filepath.c_str()); 
             
            ////////////create thread/////////////////
            
            //pthread_t id;
            //int ret;
			//ret=pthread_create(&id, NULL, thread_result_client, NULL);
			//if(ret!=0){
			//	cout<<"create pthread error"<<endl;
			//	exit(1);
			//}
			

            ///////////////////////////////////////////////
			//��ʾͼ��
			//CvFont font;
			//cvInitFont(&font,CV_FONT_VECTOR0,1,1,0,1,8);
			//��ͼ������ʾ�ı��ַ���
			//cvPutText(image,"HELLO",cvPoint(20,20),&font,CV_RGB(255,255,255));

			destroyAllWindows();
			string info = "pred: " + category;					
			imshow(info, image);
			cvWaitKey(100); // ��ͣ0.1s��ʾͼ��
			//pthread_join(id, NULL);
			cout<<endl;
		}	
	}
	cout<<"Total test image: "<<sum<<endl;
	cout<<"Correct prediction: "<<right<<endl;	
	cout<<"Accuracy: "<<(double(right)/sum)<<endl;
	return 0;
}

void getdirlist(char *path, vector<string> &dirlist){
    
     struct dirent* ent = NULL;
     DIR *pDir;
     pDir=opendir(path);
     while (NULL != (ent=readdir(pDir)))
     {
        if (ent->d_type==4)
         {
          
            dirlist.push_back(ent->d_name);
         }
     }
     return;
}

void getfilelist(char *path, vector<string> &filelist){   
     struct dirent* ent = NULL;
     DIR *pDir;
     pDir=opendir(path);   
     while (NULL != (ent=readdir(pDir)))
     {
        if (ent->d_type==8)
         {         
            filelist.push_back(ent->d_name);
         }
     }
     return;
}
