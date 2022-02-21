Requirements to compile the MLDP solution:
	-Operating system: Windows 10
	-Programming language: C++ 11
	-Additional libraries: Qt 5.12.0

To run the MLDP:
	-Operating system: Windows 10
	-All additional files are included in the solution. The executable (TestApplication.exe) can be found in Example /Bin_MLDP directory.

To run the MLDP over single/Multicenter data, three arguments are required:
	1. the settings directory path with Settings.ini and pluginSettings.ini files included (see Example directory)
	2. the dataset directory path with feature and label data fiels included. (see Example directory)
	3. SINGLE for single-center analysis or MULTI for multi-center analysis

	-Terminal line example: D:\MLDP\Example\Bin_MLDP>TestApplication.exe D:\MLDP\Example\settings\ D:\MLDP\Example\dataset\ SINGLE

-Data files structure:
	-Single center data: FDB.csv - feature dataset, LDB.csv - label dataset (See Example/dataset directory)
	-Multiple center data: TDS.csv - training feature dataset, 
			       TLD.csv - training label dataset 
			       VDS.csv - validation feature dataset
			       VLS.csv - validation label dataset
			       
	-Note: All feature names must be in A::B::C format (e.g. PET::GLCM::cm.clust.shade) -> See Example/dataset directory for tables formating