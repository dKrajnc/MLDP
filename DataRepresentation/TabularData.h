/*!
* \file This file is part of Datarepresentation module.
* The TabularData class is a general container to hold a key and its respective values in tabular form.
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#include <DataRepresentation/Export.h>
#include <DataRepresentation/Types.h>
#include <QString>
#include <QVariant>
#include <QHash>
#include <QList>
#include <QFile>
#include <QDataStream>
#include <QVector>

namespace lpmldata
{

//-----------------------------------------------------------------------------

typedef QMap< QString , QVariant >      TabularDataHeader;
typedef QHash< QString, QVariantList >  TabularDataTable;

enum class TabularDataMerge
{
	Rows = 0,
	Columns
};

//-----------------------------------------------------------------------------

/*!
* \brief Tabular data class for storing key-value list pairs.
*
* \details 
*/
class DataRepresentation_API TabularData
{

public:
	/*!
	* \brief Default constructor.
	*/
	TabularData();

	/*!
	* \brief Constructor.
	* \param [in] aName Name of the tabular data.
	*/
	TabularData( const QString& aName );

	/*!
	* \brief Copy constructor.
	* \param [in] aOther Object to copy.
	*/
	TabularData( const TabularData& aOther );

	/*!
	* \brief Move constructor.
	* \param [in] aOther Object to move.
	*/
	TabularData( TabularData&& aOther );

	/*!
	* \brief Destructor.
	*/
	~TabularData();

	/*!
	* \brief Returns with the hash representing the table itself.
	* \return The Qhash containing the table.
	*/
	TabularDataTable& table() { return mTable; }

	const TabularDataTable& table() const { return mTable; }

	/*!
	* \brief Returns with the map representing the table header.
	* \return The QMap containing the header.
	*/
	TabularDataHeader& header() { return mHeader; }

	const TabularDataHeader& header() const { return mHeader; }

	void setHeader( QStringList aHeaderNames );

	void setHeader( QString aHeaderName );

	//QMap< int, QList< QString > > headerIndexable();

	const QStringList headerNames() const;

	/*!
	* \brief Returns with the row of the respected key.
	* \param [in] aKey The key of the requested value.
	* \return The value list of the key.
	*/
	const QVariantList value( const QString& aKey ) const { return mTable.value( aKey ); }

	QVariantList& value( const QString& aKey ) { return mTable[ aKey ]; }

	QVariantList& operator[]( const QString& aKey ) { return mTable[ aKey ]; }

	lpmldata::TabularData& operator=(const lpmldata::TabularData& aRight)
	{
		mTable  = aRight.mTable;
		mHeader = aRight.mHeader;
		mName   = aRight.mName;
		return *this;
	}

	/*!
	* \brief Returns with the value of the respected key and column index.
	* \param [in] aKey The key of the requested value.
	* \param [in] aColumnIndex The column index of the requested value.
	* \return The value of the key at the given column index.
	*/
	const QVariant valueAt( QString aKey, int aColumnIndex ) const { return mTable.value( aKey ).at( aColumnIndex ); }

	QVariant& valueAt( QString aKey, int aColumnIndex ) { return mTable[ aKey ][ aColumnIndex ]; }

	/*!
	* \brief Inserts a new value identified with the key.
	* \param [in] aKey The key of the value to insert.
	* \param [in] aValue the value of the key to insert.
	*/
	void insert( const QString& aKey, const QVariantList& aValue ) { mTable.insert( aKey, aValue ); }	

	/*!
	* \brief Removes all values associated with the input key.
	* \param [in] aKey The key of the value to remove.
	* \return The number of values removed.
	*/
	int remove( const QString& aKey ) { return mTable.remove( aKey ); }

	/*!
	* \brief Returns with the unique keys located in the table.
	* \return The list of the unique keys.
	*/
	QList< QString > keys() const { return mTable.uniqueKeys(); }

	QVariantList column( unsigned int aColumnIndex ) const;

	QVariantList column( QString aColumnName );

	QString columnName( int aColumnIndex ) const { return mHeader.value( QString::number( aColumnIndex ) ).toStringList().at( 0 ); }

	/*!
	* \brief Clears the table.
	*/
	void clear() { mTable.clear(); }

	unsigned int rowCount() const { return mTable.count(); }

	unsigned int columnCount() const { return mHeader.size(); }

	const QString& name() const { return mName; }

	QString& name() { return mName; }


	/*!
	* \brief ...
	* \param [in] ...
	*/
	double mean( unsigned int aColumnIndex );

	/*!
	* \brief ...
	* \param [in] ...
	*/
	double deviation( unsigned int aColumnIndex );

	double min( unsigned int aColumnIndex );
	double max( unsigned int aColumnIndex );

	QVector< double > mins();
	QVector< double > maxs();

	QVariantList means();

	QVariantList deviations();

	void mergeRecords( lpmldata::TabularData& aTabularData );
	static lpmldata::TabularData mergeFeatures( QList< lpmldata::TabularData > aTabularDatas, TabularDataMerge aMerge );

	friend QDataStream& TabularData::operator<<( QDataStream &out, TabularData& aTabularData )
	{
		out << aTabularData.mTable
			<< aTabularData.mHeader
			<< aTabularData.mName;

		return out;
	}

	friend QDataStream& TabularData::operator>>( QDataStream &in, TabularData& aTabularData )
	{
		in  >> aTabularData.mTable
			>> aTabularData.mHeader
			>> aTabularData.mName;

		return in;
	}

	//Denis
	bool isEmpty()
	{
		if ( mTable.isEmpty() )
		{
			return true;
		}
		else
		{
			return false;
		}
	}


private:
	lpmldata::TabularDataTable   mTable;        //!< The table containing the key and a respective variant list.
	lpmldata::TabularDataHeader  mHeader;       //!< The table header containing the names and types of the columns. Key column is not taken into account.
	QString                      mName;         //!< The name of the tabular data.

};

//-----------------------------------------------------------------------------

}
