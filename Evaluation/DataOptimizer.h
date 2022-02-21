/*!
* \file
* DataOptimzier class defitition. This file is part of Evaluation module.
* The DataOptimzier class performs feature and label dataset optimization by checking the missing values, validity of features and sample keys etc.
*
* \remarks
*
* \authors
* dKrajnc
*/

#pragma once

#include <FileIo\TabularDataFileIo.h>
#include <DataRepresentation/DataPackage.h>
#include <Evaluation/Export.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

class Evaluation_API DataOptimizer
{
	DataOptimizer();

public:
	
	/*!
	* \brief Constructor to load the feature database
	* \param [in] aFDB Feature database
	*/
	DataOptimizer::DataOptimizer( const lpmldata::TabularData& aFDB )
		: mFDB( aFDB ),
		mRedundandFeatures()
	{
	}


	/*!
	* \brief Destructor
	*/
	~DataOptimizer() {};


	/*!
	* \brief Analize the loaded dataset
	*/
	void build();


	/*!
	* \brief get redundand features 
	* \return QStringList of redundand features
	*/
	QStringList redundandFeatures() const { return mRedundandFeatures; }


	/*!
	* \brief get optimized feature database
	* \return lpmldata::TabularData optimized feature database
	*/
	lpmldata::TabularData optimizedFeatureDatabase() const { return mFDB; }


private:
	QVector< double > generateMissingValues( QVector< double >& aFeatureVector ) const;
	bool hasEnoughTrueValues( const QVector< double >& aFeatureVector ) const;
	bool isRedundand( const QVector< double >& aFeatureVector ) const;
	lpmldata::TabularData featureDatabaseSubset( const QStringList& aPurifiedHeader, const lpmldata::TabularData& aFDB ) const;
	

private:
	QStringList mRedundandFeatures;
	lpmldata::TabularData mFDB;
	
};
}