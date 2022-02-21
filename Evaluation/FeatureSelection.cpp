#include <Evaluation/FeatureSelection.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

void FeatureSelection::build( const lpmldata::DataPackage& aDataPackage )
{
	// Create switch which will call rank function e.g. rSquaredRank, RF, etc.	
	if ( mRankMethod == "RSquared" )
	{
		mFeatureRanks = rSquaredRank( aDataPackage );
	}
	else
	{
		qDebug() << "SFS - Error: Ranking method is not defined! ";
	}

	int correctedFeatureCount        = 0;
	int forwardSelectionFeatureCount = std::max( 2, mFeatureCount );
	correctedFeatureCount            = std::min( forwardSelectionFeatureCount, aDataPackage.featureCount() );

	for ( int i = 0; i < correctedFeatureCount; ++i )
	{
		double max = -DBL_MAX;
		QString maxFeatureName;

		for ( QMap< QString, double >::iterator it = mFeatureRanks.begin(); it != mFeatureRanks.end(); ++it )
		{
			auto key   = it.key();
			auto value = it.value();		

			if ( mSelectedFeatures.contains( key ) ) continue;			

			if ( value > max )
			{
				max            = value;
				maxFeatureName = key;
			}
		}

		mSelectedFeatures.push_back( maxFeatureName );
	}	
}

//-----------------------------------------------------------------------------

lpmldata::DataPackage FeatureSelection::run( const lpmldata::DataPackage& aDataPackage )
{
	
	if ( !mIsInitValid )
	{
		lpmldata::DataPackage result( aDataPackage.featureDatabase(), aDataPackage.labelDatabase() );
		return result;
	}
	// Create new dataset
	lpmldata::TabularData updatedFDB = aDataPackage.featureDatabaseSubset( mSelectedFeatures );	

	lpmldata::DataPackage result( updatedFDB, aDataPackage.labelDatabase() );
	return result;
}

//-----------------------------------------------------------------------------

QMap< QString, double > FeatureSelection::rSquaredRank( const lpmldata::DataPackage& aDataPackage )
{
	QMap< QString, double > ranks;	

	auto activeIndex = aDataPackage.activeLabelIndex();
	auto labelGroups = aDataPackage.labelGroups();
	auto commonKeys  = aDataPackage.commonKeys();	
	auto FDB         = aDataPackage.featureDatabase();
	auto LDB         = aDataPackage.labelDatabase();
	auto headers     = FDB.headerNames();

	
	// Read out the label values and feature values from the dataset.
	for ( int i = 0; i < headers.size(); ++i )
	{
		QVector< double > x;
		QVector< double > y;
		
		for ( int j = 0; j < commonKeys.size(); ++j )
		{
			auto key = commonKeys.at( j );

			double featureValue = FDB.valueAt( key, i ).toDouble();
			auto   label        = LDB.valueAt( key, activeIndex ).toString();
			double labelValue   = labelGroups.indexOf( label ); 

			x.push_back( featureValue );
			y.push_back( labelValue ); // Create a numerical value list for labels.
		}
		
		// Calculate the mean values of feature and label values separately
		double sumOfElementsX = 0.0;
		double sumOfElementsY = 0.0;
		double denominatorX   = x.size();
		double denominatorY   = y.size();

		for ( double& element : x )
		{
			sumOfElementsX += element;
		}

		for ( double& element : y )
		{
			sumOfElementsY += element;
		}

		double meanFeature = sumOfElementsX / denominatorX;
		double meanLabel   = sumOfElementsY / denominatorY;

		// Calculate the difference between actual feature value and mean feature value
		QVector< double > substractedX;
		QVector< double > substractedY;

		for ( double& element : x )
		{
			substractedX.push_back( element - meanFeature );
		}

		// Calculate the difference between actual label value and mean label value
		for ( double& element : y )
		{
			substractedY.push_back( element - meanLabel );
		}

		// Calculate the squared difference of actual feature value and mean feature value
		// and sum the results 
		QVector< double > substractedXSquared;	

		double sumOfSquaredElementsX = 0;
	
		for ( double& element : substractedX )
		{
			substractedXSquared.push_back( pow( element, 2.0 ) );
		}

		for ( double& element : substractedXSquared )
		{
			sumOfSquaredElementsX += element;
		}

		// Calculate the squared difference of actual label value and mean label value
		// and sum the results
		QVector< double > substractedYSquared;

		double sumOfSquaredElementsY = 0;

		for ( double& element : substractedY )
		{
			substractedYSquared.push_back( pow( element, 2.0 ) );
		}

		for ( double& element : substractedYSquared )
		{
			sumOfSquaredElementsY += element;
		}
	
		// Calculate the difference between actual feature value and mean feature value
		// multiplied with the difference between actual label value and mean label value
		// and sum the results
		QVector< double > slopeNumerator;

		double sumOfSlopeNumerators = 0;

		for ( int i = 0; i < x.size(); ++i )
		{
			slopeNumerator.push_back( substractedX.at( i ) * substractedY.at( i ) );
		}
	
		for ( double& element : slopeNumerator )
		{
			sumOfSlopeNumerators += element;
		}
		
		double slope     = sumOfSlopeNumerators / sumOfSquaredElementsX;  // Calculate the slope of the regression line
		double intersect = -( slope * meanFeature ) + meanLabel;  // Calculate the Y intersect point of the regression line

		// Calculate points of the regression line for each feature value 
		QVector< double > estimatedValues;

		for ( int i = 0; i < x.size(); ++i )
		{
			estimatedValues.push_back( intersect + ( slope * x.at( i ) ) );
		}
	
		// Calculate the squared difference between feature values on regression line and mean value 
		// of label values and sum the results
		QVector< double > substractedEstimated;
		QVector< double > substractedEstimatedSquared;

		double sumOfsubstractedEstimatedSquared = 0;

		for ( double& element : estimatedValues )
		{
			substractedEstimated.push_back( element - meanLabel );
		}

		for ( double& element : substractedEstimated )
		{
			substractedEstimatedSquared.push_back( pow( element, 2.0 ) );
		}

		for ( double& element : substractedEstimatedSquared )
		{
			sumOfsubstractedEstimatedSquared += element;
		}

		double rSquared = sumOfsubstractedEstimatedSquared / sumOfSquaredElementsY;  // Calculate rSquared		
			
		ranks.insert( headers.at( i ), rSquared );
	}

	return ranks;
}

}
