/*!
* \file
* DataPackage class defitition. This file is part of DataRepresentation module.
* The DataPackage is a class for describing feature and label database. It contains basic feature and sample information such as counts, indices, keys, labels, euclidean distances etc.
*
* \remarks
*
* \authors
* dKrajnc
*/

#pragma once

#include <DataRepresentation/Export.h>
#include <DataRepresentation/TabularData.h>
#include <QDebug>
#include <QString>
#include <QList>

namespace lpmldata
{

class DataRepresentation_API DataPackage
{

public:
	DataPackage( lpmldata::TabularData aFDB, lpmldata::TabularData aLDB )
	:
		mFDB( aFDB ), 
		mLDB( aLDB ),
		mActiveLabelIndex( 0 ),
		mLabelIndex(),
		mLabelOutcomes(),
		mLabelName(),
		mSampleKeys(),
		mIsValidDataset(),
		mFeatureCount(),
		mIncludedKeys()
	{
		mLabelName = mLDB.headerNames().at( 0 );
		initialize( aFDB, aLDB, mLabelName );


	}

	DataPackage( lpmldata::TabularData& aFDB, lpmldata::TabularData& aLDB, QString aLabelName, QStringList aIncludedKeys = {} );

	const lpmldata::TabularData& featureDatabase() const
	{
		return mFDB;
	}

	const lpmldata::TabularData& labelDatabase() const
	{
		return mLDB;
	}

	lpmldata::TabularData& featureDatabase()
	{
		return mFDB;
	}

	lpmldata::TabularData& labelDatabase()
	{
		return mLDB;
	}

	void setActiveLabel( QString aLabel );

	void setActiveLabelIndex( int aIndex ) { mActiveLabelIndex = aIndex; };

	const QStringList labelGroups() const;	

	QStringList commonKeys() const;

	int activeLabelIndex() const
	{
		return mActiveLabelIndex;
	}

	int featureCount() const
	{
		return mFDB.columnCount();
	}

	const QString& labelName() const { return mLabelName; }
	const int labelIndex() const { return mLabelIndex; }
	const QStringList labelOutcomes() { return mLabelOutcomes; }
	const QStringList sampleKeys() { return mSampleKeys; }
	const bool isValid() { return mIsValidDataset; }
	bool isBalanced();

	//Store FDB
	//In feature space
	lpmldata::TabularData featureDatabaseSubset( QStringList aFeatureNames ) const; //Used in: FeatureSelection
	lpmldata::TabularData featureDatabaseSubset( QVector< QVector< double > >& aFeatureValues, QStringList& aFeatureNames ) const; //Used in PCA
	lpmldata::TabularData featureDatabaseSubset( QVector< QVector< double > >& aFeatureColumns ) const;
	//In sample space
	lpmldata::TabularData sampleDatabaseSubset( const QStringList& aKeys ) const; //Used in: IsolationForest, TomekLinks, RandomUndersampling
	lpmldata::TabularData sampleDatabaseSubset( QMap< QString, QVector< double > >& aSynthSamples ) const; //Used in: Oversampling
	lpmldata::TabularData syntheticSampleDatabaseSubset( QMap< QString, QVector< double > >& aSynthSamples ) const;

	//Store LDB 
	//When keys and labels are generated
	lpmldata::TabularData labelDatabaseSubset( QStringList& aSynthNames, int& aMinorityLabel ) const; //Used in: Oversampling
	lpmldata::TabularData syntheticLabelDatabaseSubset( const QList< QPair< QString, double > >& aSyntheticPairList ) const;
	//When keys are selected
	lpmldata::TabularData labelDatabaseSubset( const QStringList& aKeys ) const; //Used in IsolationForest, TomekLinks, RandomUndersampling
	

	QStringList keysOfLabelGroup( QString aLabelGroup ) const;
	QMap< QString, QStringList > keysOfLabelGroups() const;

	QMap< QString, double > distance( const QMap< QString, QVariantList >& aFirst, const QMap< QString, QVariantList >& aSecond ) const;	
	double distance( QVector< double >& aFirst, QVector< double >& aSecond );
	double distance( const QVariantList& aFirst, const QVariantList& aSecond ) const;

	//lpmldata::TabularData normalize( const lpmldata::TabularData& aFDB ) const;
	double mean( const double& aSum, const int& aColumnSize ) const;
	double standardDeviation( const QVector< QVariant >& aFeatureColumn, const double& aMean ) const;
	double standardDeviation( const QVector< double >& aFeatureColumn, const double& aMean ) const;

	QVariantList featureVector( QString aKey ) const { return mFDB.value( aKey ); }
	QVector< double > featureColumn( QString aFeatureKey ) const;
	QList< int > labels( QString aLabelName ) const;
	QVector< double > normalizeFeature( const QVector< double >& aFeatureColumn ) const;
	QVariantList toQVariantList( const QVector< double >& aVector ) const;
	lpmldata::TabularData normalizeData();
	lpmldata::DataPackage normalizeDataPackage();

	void initialize( lpmldata::TabularData& aFDB, lpmldata::TabularData& aLDB, QString aLabelName );

	int minorityCount() const;
	int majorityCount() const;
	int sampleCountOfPercentage( const double& aValidationPercentage ) const;

	int getMinorityIndex() const;
	int getMinorityCount() const;
	int getMinorityLabel() const;
						   
	int getMajorityIndex() const;
	int getMajorityCount() const;
	int getMajorityLabel() const;

	QStringList getMinorityKeys() const;
	QStringList getMajorityKeys() const;
	//-----------------------------------------------------------------
	//From lpmleval::TabularDataFilter

	lpmldata::TabularData subTableByKeys( const lpmldata::TabularData& aTabularData, QStringList aReferenceKeys );
	void eraseIncompleteRecords( lpmldata::TabularData& aFeatureDatabase );
	QStringList labelGroups( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded = false );
	QStringList commonKeys( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded = false );
	QStringList keysByLabelGroup( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel );
	void updateLDB();

private:
	lpmldata::TabularData  mFDB;
	lpmldata::TabularData  mLDB;
	int                    mActiveLabelIndex;
	int                    mLabelIndex;
	QList< QString >       mLabelOutcomes;
	QString                mLabelName;
	QList< QString >       mSampleKeys;
	bool                   mIsValidDataset;
	int                    mFeatureCount;
	QStringList            mIncludedKeys;
};

}
