/*!
* \file This file is part of FileIo module.
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#include <FileIo/Export.h>
#include <DataRepresentation/TabularData.h>

namespace lpmlfio
{

//-----------------------------------------------------------------------------


class FileIo_API TabularDataFileIo
{

public:

	TabularDataFileIo();

	TabularDataFileIo( QString aWorkingDirectory );

	~TabularDataFileIo();

	void load( QString aHeaderFileName, lpmldata::TabularData& aTabularData );

	void save( QString aHeaderFileName, lpmldata::TabularData& aTabularData );

private:

	QString mWorkingDirectory;  //!< The working directory of the file tabular data file IO.
	char mCommaSeparator;       //!< Comma separator character for CSV file handling.

};

//-----------------------------------------------------------------------------

}
