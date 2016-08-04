#include "match.h"

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