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

using namespace std;

void MakeDir(const string& filepath);
void ListDir(const string& directory, bool (*filter)(const WIN32_FIND_DATA& entry), vector<string>* entries);
void GetDirList(const string& directory, vector<string>* dirlist);
void GetFileList(const string& directory, vector<string>* filelist);