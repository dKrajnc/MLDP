/*!
* \file 
*
* \remarks
*/

#include <FileIo/TabularDataFileIo.h>
#include <QFile>
#include <QTextStream>
#include <QJsonObject>
#include <QJsonDocument>
#include <QDebug>

namespace lpmlfio
{

//-----------------------------------------------------------------------------

TabularDataFileIo::TabularDataFileIo()
:
	mWorkingDirectory( "" ),
	mCommaSeparator( ';' )
{
}

//-----------------------------------------------------------------------------

TabularDataFileIo::TabularDataFileIo( QString aWorkingDirectory )
	:
	mWorkingDirectory( aWorkingDirectory ),
	mCommaSeparator( ';' )
{
}

//-----------------------------------------------------------------------------

TabularDataFileIo::~TabularDataFileIo()
{
}

//-----------------------------------------------------------------------------

void TabularDataFileIo::load( QString aHeaderFileName, lpmldata::TabularData& aTabularData )
{

	QString fullPath;
	if ( mWorkingDirectory.count() == 0 )
	{
		fullPath = aHeaderFileName;
	}
	else
	{
		fullPath = mWorkingDirectory + "/" + aHeaderFileName;
	}
	
	//Load CSV
	if ( !fullPath.contains( ".csv" ) )
	{
		fullPath = fullPath + ".csv";
	}
	QFile fileInCsv( fullPath );
	if ( fileInCsv.open( QIODevice::ReadOnly | QIODevice::Text ) )
	{
		//qDebug() << "Loading: " << fullPath + ".csv" << "...";

		QString key;
		QVariantList row;

		// Read the header
		QByteArray line = fileInCsv.readLine().trimmed();  //Read the header data which is the first line.
		QList< QByteArray > separatedLine = line.split( mCommaSeparator );  //Separate column values.

		if ( QString( separatedLine.at( separatedLine.size() - 1 ) ).compare("") == 0 )
		{
			separatedLine.removeLast();
		}
		
		// Remove possible empty strings from end.

		lpmldata::TabularDataHeader header;

		for ( int headerIndex = 1; headerIndex < separatedLine.size(); ++headerIndex )
		{
			QString value = separatedLine.at( headerIndex );
			QString type = "Float";
			QVariantList headerValue = { value, type };
			header.insert( QString::number( headerIndex - 1), headerValue );
		}

		aTabularData.header() = header;  // Save the header to the tabular data.

		// Read up the file.
		while ( !fileInCsv.atEnd() )
		{
			
			QByteArray line = fileInCsv.readLine().trimmed();  // Read one line, remove whitespaces.
			QList< QByteArray > separatedLine = line.split( mCommaSeparator );  // Separate column values.
			key = separatedLine.at( 0 );  // Read out the key.

			// Fill up the row with values of the given key.
			row.clear();
			for ( int columnIndex = 1; columnIndex < separatedLine.size(); ++columnIndex )
			{
				row.push_back( separatedLine.at( columnIndex ) );
			}

			// Place the key-value pairs into the tabular data.
			aTabularData.insert( key, row );
		}

		fileInCsv.close();

		//qDebug() << "Loading finished." << endl;
	}
	else qDebug() << "Failed to open: " << fullPath << endl;
}

//-----------------------------------------------------------------------------

void TabularDataFileIo::save( QString aFileName, lpmldata::TabularData& aTabularData )
{
	QString fullPath;
	if ( mWorkingDirectory.count() == 0 )
	{
		fullPath = aFileName;
	}
	else
	{
		fullPath = mWorkingDirectory + "/" + aFileName;
	}

	// Save CSV
	if ( !fullPath.contains( ".csv" ) )
	{
		fullPath = fullPath + ".csv";
	}
	QFile fileOutCsv( fullPath );
	if ( fileOutCsv.open( QFile::WriteOnly | QFile::Text ) )
	{
		QTextStream stream( &fileOutCsv );

		// Save header.
		QMap< QString, QVariant > header = aTabularData.header();

		stream << "Key";

		for ( int headerIndex = 0; headerIndex < header.size(); ++headerIndex )
		{
			QStringList headerColumnDescriptor = header.value( QString::number( headerIndex ) ).toStringList();
			stream << mCommaSeparator << headerColumnDescriptor.at( 0 );
		}

		stream << endl;
		fileOutCsv.flush();
		

		// Save the tabular data entries.

		auto keys = aTabularData.keys();
		std::sort( keys.begin(), keys.end() );

		for ( auto key : keys )
		{
			stream << key;
			for ( int columnIndex = 0; columnIndex < aTabularData.columnCount(); ++columnIndex )
			{
				auto cell = aTabularData.valueAt( key, columnIndex ).toString();
				if ( cell == "nan" )
				{
					cell = "NA";
				}
				stream << mCommaSeparator << cell;
			}
			stream << endl;
			fileOutCsv.flush();
		}

		fileOutCsv.close();
	}
	else
	{
		qDebug() << "Cannot open for read: " << fullPath;
	}
}

//-----------------------------------------------------------------------------

}
