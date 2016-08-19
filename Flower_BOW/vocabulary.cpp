#include "vocabulary.h"

/*
 * loop through every directory 
 * compute each image's keypoints and descriptors
 * train a dictionary
 */
Mat BuildVocabulary(const string& databaseDir, 
					const vector<string>& flowerCategories, 
					const string& vocabularyFile,
					const Ptr<FeatureDetector>& detector, 
					const Ptr<DescriptorExtractor>& extractor,
					int wordCount)
{
	Mat vocabulary; 
	FileStorage fs(vocabularyFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["vocabulary"] >> vocabulary;
	}
	else
	{
		Mat allDescriptors;
		for (int index = 0; index != flowerCategories.size(); ++index)
		{
			cout << "Processing category " << flowerCategories[index] << endl;
			string currentCategory = databaseDir + '\\' + flowerCategories[index];
			vector<string> filelist;
			GetFileList(currentCategory, &filelist);
			for (auto fileindex = filelist.begin(); fileindex != filelist.end(); fileindex += 5)
			{			
				string filepath = currentCategory + '\\' + *fileindex;
				Mat image = imread(filepath);
				if (image.empty())
				{
					continue; // maybe not an image file
				}
				vector<KeyPoint> keyPoints;
				Mat descriptors;
				detector->detect(image, keyPoints);
				extractor->compute(image, keyPoints, descriptors);
				if (allDescriptors.empty())
				{
					allDescriptors.create(0, descriptors.cols, descriptors.type());
				}
				allDescriptors.push_back(descriptors);
			}
		}
		assert(!allDescriptors.empty());
		cout << "build vocabulary..." << endl;
		BOWKMeansTrainer bowTrainer(wordCount);
		vocabulary = bowTrainer.cluster(allDescriptors);
		fs.open(vocabularyFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			//将vocabulary矩阵保存在fs对象指定的xml文件的vocabulary标签下
			fs << "vocabulary" << vocabulary;
		}
		cout << "done build vocabulary..." << endl;
	}	
	return vocabulary;
}