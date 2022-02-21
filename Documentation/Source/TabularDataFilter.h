/*!
* \file
* TabularDataFilter class defitition. This file is part of Evaluation module.
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#include <Evaluation/Export.h>
#include <DataRepresentation/DataPackage.h>
//#include <DataRepresentation/TabularData.h>
#include <QDebug>
#include <random>

namespace lpmleval
{

//-----------------------------------------------------------------------------

enum class FoldSelector
{
	LOO = 0,
	KFold,
	MCEqual,
	MCStratified
};

class Evaluation_API TabularDataFilter
{

public:

	TabularDataFilter();

	virtual ~TabularDataFilter() {}

	QVector< QMap< QVariant, double > > equalizedBagging( lpmldata::DataPackage* aDataPackage, const unsigned int aNumberBags, const double aBagFraction );


	//-----------------------------------------------------------------------------
	// Added by cs

	//QVector< QMap< QVariant, double > > stratifiedBagging( lpmldata::TabularData& aLabelSet, unsigned int aNumberBags, const double aBagFraction )
	//{
	//	QVector< QMap< QVariant, double > > keysToWeights;

	//	// Create subsets of original keys depending on their label
	//	QMap< QVariant, QVector< QString > > keysByLabel;

	//	for ( auto label : aLabelSet.uniqueValues( 0 ) )
	//	{
	//		keysByLabel[ label ] = {};
	//	}

	//	for ( auto key : aLabelSet.keys() )
	//	{
	//		QVariant label = aLabelSet.valueAt( key, 0 );
	//		keysByLabel[ label ].append( key );
	//	}

	//	while ( aNumberBags > 0 )
	//	{
	//		QMap< QVariant, double > bagMap;		// Keys to weights for one bag

	//		// Draw from subsets with uniform labels
	//		for ( auto labelSubSet : keysByLabel.values() )
	//		{
	//			for ( int drawnSamples = 0; drawnSamples < floor( labelSubSet.size() * aBagFraction ); drawnSamples++ )
	//			{
	//				QString sampledKey = sample( labelSubSet );

	//				bool keyExists;
	//				if ( bagMap.keys().contains( sampledKey ) )	keyExists = true;

	//				if ( keyExists )	bagMap[ sampledKey ] += 1;
	//				else				bagMap[ sampledKey ] = 1;
	//			}
	//		}

	//		keysToWeights.append( bagMap );
	//		aNumberBags--;
	//	}

	//	return keysToWeights;
	//}

	//-----------------------------------------------------------------------------

	QString sample( const QVector< QString >& aVector )
	{
		std::random_device rd;
		std::mt19937 engine{ rd() };
		std::uniform_int_distribution<int> dist( 0, aVector.size() - 1 );

		return aVector[ dist( engine ) ];
	}

	//-----------------------------------------------------------------------------
	// Added by cs

	QVector< QMap< QVariant, double > > bagging( lpmldata::DataPackage* aDataPackage, const unsigned int aNumberBags, const double aBagFraction );
	

	//-----------------------------------------------------------------------------
	// Added by cs

	QVector< QMap< QVariant, double > > walkerBagging( lpmldata::TabularData& aLabelSet, unsigned int aNumberBags, const double aBagFraction );

	//-----------------------------------------------------------------------------

	QMap< QVariant, double > resampleWithWeights( lpmldata::TabularData& aLabelSet, QMap< QVariant, double >& aBagWeights, bool aRepresentUsingWeights );

	//-----------------------------------------------------------------------------

	void normalize( QVector< double >& aVector );


	//-----------------------------------------------------------------------------
	// Added by cs

	lpmldata::TabularData subTableByAttributes( lpmldata::TabularData& aFeatureSet, const QList< int >& aAttributeIndices );
	
	//-----------------------------------------------------------------------------


	void eraseIncompleteRecords( lpmldata::TabularData& aFeatureDatabase );

	lpmldata::TabularData subtableByKeyExpression( lpmldata::TabularData& aFeatureDatabase, QString aKeyExpression );

	lpmldata::TabularData subtableByReductionCovaraince( lpmldata::TabularData& aFeatureDatabase, lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, double aSimilarityThreshold );

	lpmldata::TabularData subtableByReductionKDE( lpmldata::TabularData& aFeatureDatabase, lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aFeatureCount );

	lpmldata::TabularData subTableByLabelGroup( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel );

	QStringList keysByLabelGroup( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const QString aReferenceLabel );

	lpmldata::TabularData bootstrap( const lpmldata::TabularData& aFeatureDatabase, const ulint aSampleCount, bool aIsOriginalSamplesPreserved = false );

	lpmldata::TabularData bootstrapN( const lpmldata::TabularData& aFeatureDatabase );

	QPair< lpmldata::TabularData, lpmldata::TabularData> bootstrapN( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase );

	QPair< lpmldata::TabularData, lpmldata::TabularData> bootstrap( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const ulint aSampleCount, bool aIsOriginalSamplesPreserved = false );

	lpmldata::TabularData subTableByUniqueness( const lpmldata::TabularData& aFeatureDatabase );

	inline bool isEqual( const QVariantList& aFirst, const QVariantList& aSecond )
	{

		double difference = 0.0;

		for ( int vectorIndex = 0; vectorIndex < aFirst.size(); ++vectorIndex )
		{
			difference += std::abs( aFirst.at( vectorIndex ).toDouble() - aSecond.at( vectorIndex ).toDouble() );
		
		}

		if ( difference / aFirst.size() < DBL_EPSILON )
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	
	QStringList labelGroups( const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded = false );

	lpmldata::TabularData subTableByKeys( const lpmldata::TabularData& aTabularData, QStringList aReferenceKeys );

	QStringList commonKeys( const lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, bool aIsNAIncluded = false );

	lpmldata::TabularData normalize( lpmldata::TabularData& aFeatureDatabase );

	lpmldata::TabularData subTableByMask( const lpmldata::TabularData& aFeatureDatabase, const QVector< double >& aFeatureMask );

	lpmldata::TabularData subTableByFeatureNames( const lpmldata::TabularData& aFeatureDatabase, const QStringList& aFeatureNames );

	QVector< int > maskIndicesByNames( const lpmldata::TabularData& aFeatureDatabase, const QStringList& aFeatureNames );

	double distance( const QVariantList& aFirstVector, const QVariantList& aSecondVector, QVector< double > aFeatureMask = {} );

	QStringList nearestNeighbors( const QString& aKey, lpmldata::TabularData& aFeatureDatabase, int aNeighborCount, QVector< double > aFeatureMask = {} );

	QStringList nearestNeighbors( const QVariantList& aFeatureVector, lpmldata::TabularData& aFeatureDatabase, int aNeighborCount, QVector< double > aFeatureMask = {} );

	QMap< QString, QStringList > nearestNeighborMap( lpmldata::TabularData& aFeatureDatabase, int aNeighborCount, QVector< double > aFeatureMask = {} );

	QMap< QString, double > scoreFeaturesByMSMOTE( lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aNeighborCount, QVector< double > aFeatureMask = {} );

	QMap< QString, QString > categorizeMSMOTE( const QMap< QString, double >& aMSMOTEScores, double aOutlierThreshold, double aSafeThreshold );

	int MSMOTECategoryCount( const QMap< QString, QString >& aMSMOTECategories, QString aCategoryType, QStringList aReferenceKeys = {} );

	QPair< lpmldata::TabularData, lpmldata::TabularData> MSMOTE( lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, const ulint aSampleCount, QVector< double > aFeatureMask = {} );

	QMap< int, QStringList > stratifiedKFoldMap(   int aFoldCount, double aSubsetRatio, lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex );

	lpmldata::TabularData foldConfigTable( int aFoldCount, double aSubsetRatio, lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex );

	lpmldata::TabularData foldConfigTableGenerator( int aFoldCount, double aSubsetRatio, lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, FoldSelector aFoldSelector );
	
	lpmldata::TabularData leaveOneOutSubSampler(              lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex );	
	lpmldata::TabularData stratifiedKFoldSubSampler(          lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aFoldCount );	
	lpmldata::TabularData ballancedPermutationSubSampler(     lpmldata::TabularData& aFeatureDatabase, const lpmldata::TabularData& aLabelDatabase, const int aLabelIndex, int aMaximumFoldCount, double aSubsetRatio );
	
	

	//-----------------------------------------------------------------------------


private:

	std::mt19937_64 mRandomGenerator; 
	QString mAbc;
};

}
