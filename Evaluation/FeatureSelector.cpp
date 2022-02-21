#include <Evaluation/FeatureSelector.h>
#include <QDebug>

namespace lpmleval
{

//-----------------------------------------------------------------------------

FeatureSelector::FeatureSelector( lpmldata::TabularData aFeatureDatabase, lpmldata::TabularData aLabelDatabase, int aLabelIndex )
:
	mFeatureDatabase( aFeatureDatabase ),
	mLabelDatabase( aLabelDatabase ),
	mLabelIndex( aLabelIndex ),
	mFeatureCount( aFeatureDatabase.columnCount() ),
	mRd(),
	mEng( mRd() ),
	mRealDistribution( 0.0, 1.0 )
{
}

//-----------------------------------------------------------------------------

FeatureSelector::~FeatureSelector()
{

}

//-----------------------------------------------------------------------------

QVector< double > FeatureSelector::executeFrCovMxGlobal( double aCovarianceThreshold )
{
	lpmleval::CovarianceMatrix covMatrix( mFeatureDatabase );  // Build up covariance matrix based on the feature vector database.
	//qDebug() << "CovMx created";
	QVector< double > featureDeviations;
	featureDeviations.resize( mFeatureDatabase.columnCount() );

	// Collect feature deviations for speed optimization.
	for ( int columnIndex = 0; columnIndex < mFeatureDatabase.columnCount(); ++columnIndex )
	{
		//featureDeviations.push_back( mFeatureDatabase.deviation( columnIndex ) );
		featureDeviations[ columnIndex ] = mFeatureDatabase.deviation( columnIndex );
	}

	// Initial containers for previous and new feature masks.
	QVector< double > masks;
	masks.resize( mFeatureCount );
	masks.fill( 1.0 );  // Initialize the feature mask.

	// Maintain a feature mask containing who was already filtered out as redundant one.
	QVector< double > filteredFeaturesMask;
	filteredFeaturesMask.resize( mFeatureCount );
	filteredFeaturesMask.fill( 0.0 );

	// Select those features that have a more or equal correlation than the threshold. Select only one of these which has the largest deviation
	for ( int i = 0; i < covMatrix.rowCount(); ++i )
	{
		//qDebug() << i;
		if ( filteredFeaturesMask.at( i ) == 1.0 )
		{
			continue; // Have we filtered this guy out already? If yes, then skip it.
		}

		for ( int j = 0; j < covMatrix.columnCount(); ++j )
		{
			if ( i == j ) continue;  // We do not filter out ourselves.

			if ( filteredFeaturesMask.at( j ) == 1.0 )
			{
				continue; // Have we filtered this guy out already? If yes, then skip it.
			}

			double firstDev = featureDeviations.at( i );
			double secondDev = featureDeviations.at( j );

			// TODO: Remove 100% correlating features (e.g. volume)

			if ( std::abs( covMatrix( i, j ) ) > aCovarianceThreshold )  // We found redundant feature pairs.
			{		
				if ( firstDev > secondDev )  // Keep the first, delete the second;
				{
					masks[ j ] = 0.0;
					filteredFeaturesMask[ j ] = 1.0;
				}
				else  // Keep the second, delete the first;
				{
					masks[ i ] = 0.0;
					filteredFeaturesMask[ i ] = 1.0;
				}
			}
		}
	}

	return masks;
}

//-----------------------------------------------------------------------------

QVector< double > FeatureSelector::executeFrCovMxLocal( double aCovarianceThreshold )
{

	lpmleval::CovarianceMatrix covMatrix( mFeatureDatabase );  // Build up covariance matrix based on the feature vector database.
	QVector< double > featureDeviations;

	// Collect feature deviations for speed optimization.
	for ( int columnIndex = 0; columnIndex < mFeatureDatabase.columnCount(); ++columnIndex )
	{
		featureDeviations.push_back( mFeatureDatabase.deviation( columnIndex ) );
	}

	// Initial containers for previous and new feature masks.
	QVector< double > masks;
	masks.resize( mFeatureCount );
	masks.fill( 1.0 );  // Initialize the feature mask.

	// Maintain a feature mask containing who was already filtered out as redundant one.
	QVector< double > filteredFeaturesMask;
	filteredFeaturesMask.resize( mFeatureCount );
	filteredFeaturesMask.fill( 0.0 );

	// Select those features that have a more or equal correlation than the threshold. Select only one of these which has the largest deviation
	for ( int i = 0; i < covMatrix.rowCount(); ++i )
	{
		if ( filteredFeaturesMask.at( i ) == 1.0 )
		{
			continue; // Have we filtered this guy out already? If yes, then skip it.
		}

		for ( int j = 0; j < covMatrix.columnCount(); ++j )
		{
			if ( i == j ) continue;

			if ( filteredFeaturesMask.at( j ) == 1.0 )
			{
				continue; // Have we filtered this guy out already? If yes, then skip it.
			}

			double firstDev = featureDeviations.at( i );
			double secondDev = featureDeviations.at( j );

			if ( std::abs( covMatrix( i, j ) ) > aCovarianceThreshold )  // We found redundant feature pairs.
			{
				// Read out feature categories.
				QString firstModality = mFeatureDatabase.columnName( i ).split( "::" ).at( 0 );
				QString firstCategory = mFeatureDatabase.columnName( i ).split( "::" ).at( 1 ).split( "::" ).at( 0 );

				QString secondModality = mFeatureDatabase.columnName( j ).split( "::" ).at( 0 );
				QString secondCategory = mFeatureDatabase.columnName( j ).split( "::" ).at( 1 ).split( "::" ).at( 0 );

				// If correlating features are from the same modality and feature category...
				if ( firstModality == secondModality && firstCategory == secondCategory )  // Same modalities and feature categories?
				{
					if ( firstDev > secondDev )  // Keep the first, delete the second;
					{
						masks[ j ] = 0.0;
						filteredFeaturesMask[ j ] = 1.0;
					}
					else  // Keep the second, delete the first;
					{
						masks[ i ] = 0.0;
						filteredFeaturesMask[ i ] = 1.0;
					}
				}	
			}
		}
	}

	return masks;
}

//-----------------------------------------------------------------------------

QVector< double > FeatureSelector::executeFsRandom( double aChanceToSelect )
{
	// Initialize the mask vector.
	QVector< double > masks;
	masks.resize( mFeatureCount );
	masks.fill( 0.0 );	

	// Select features randomly.
	for ( int featureIndex = 0; featureIndex < mFeatureCount; ++featureIndex )
	{
		if ( mRealDistribution( mEng ) < aChanceToSelect ) masks[ featureIndex ] = 1.0;
	}

	return masks;
}

//-----------------------------------------------------------------------------

QVector< double > FeatureSelector::executeFsKdeOverlap( int aSelectedFeatureCount, int aBootstrapSize )
{
	KernelDensityExtractor Kde( mFeatureDatabase, mLabelDatabase, mLabelIndex );

	// Read out the overlap ratios of features from the KDE.
	QMap< double, int > overlapRatios = Kde.overlapRatios();

	// Initialize the mask vector.
	QVector< double > masks;
	masks.resize( mFeatureCount );
	masks.fill( 0.0 );

	// Select the first aSelectedFeatureCount features.
	QMap< double, int >::const_iterator it;
	//QList< double > keys = overlapRatios.keys();
	//for ( int keyIndex = 0; keyIndex < aSelectedFeatureCount; ++keyIndex )
	it = overlapRatios.begin();
	for ( int i = 0; i < std::min( aSelectedFeatureCount, overlapRatios.size() ); ++i )
	{
		double overlapRatio = it.key();
		int selectedFeature = it.value();
		masks[ selectedFeature ] = 1.0;
		++it;
	}

	return masks;
}

//-----------------------------------------------------------------------------

}
