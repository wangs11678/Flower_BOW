#include "utils.h"

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