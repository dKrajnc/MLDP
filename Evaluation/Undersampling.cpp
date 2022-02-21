#include <Evaluation/Undersampling.h>


namespace dkeval
{

//-----------------------------------------------------------------------------

void Undersampling::build( const lpmldata::DataPackage& aDataPackage )
{
	if ( mType == "RandomUndersampling" )
	{
		randomUndersampling( aDataPackage );
	}
	else if ( mType == "TomekLinks" )
	{
		tomekLinks( aDataPackage );
	}
}

//-----------------------------------------------------------------------------

lpmldata::DataPackage Undersampling::run( const lpmldata::DataPackage& aDataPackage )
{
	auto keys         = aDataPackage.commonKeys();
	auto purifiedKeys = keys.toSet().subtract( mChoosenSamples.toSet() ).toList();

	//Create new database
	auto updatedFDB = aDataPackage.sampleDatabaseSubset( purifiedKeys );
	auto updatedLDB = aDataPackage.labelDatabaseSubset( purifiedKeys );

	lpmldata::DataPackage result( updatedFDB, updatedLDB );

	return result;
}

//-----------------------------------------------------------------------------

int Undersampling::counterDeterminant( const QVariantList& aFeature, QString aFeatureName, const QVariantList& aMinorityFeature, QString aMinorityName, const QVariantList& aMajorityFeature, QString aMajorityName, const double& aDistance )
{
	int counter = 0;

	if ( aMinorityName != aFeatureName && aMajorityName != aFeatureName )
	{
		auto minorityDistance = distance( aMinorityFeature, aFeature );				
		auto majorityDistance = distance( aMajorityFeature, aFeature );

		if ( aDistance < minorityDistance || aDistance < majorityDistance )
		{
			counter++;
		}
	}

	return counter;
}

//-----------------------------------------------------------------------------

double Undersampling::distance( const QVariantList& aFirst, const QVariantList& aSecond )
{
	double distance           = 0.0;
	double substractedSquared = 0.0;

	for ( int i = 0; i < aFirst.size(); ++i )
	{
		substractedSquared = pow( aSecond.at( i ).toDouble() - aFirst.at( i ).toDouble(), 2.0 );
		substractedSquared += substractedSquared;
	}
	distance = sqrt( substractedSquared );

	return distance;
}

//-----------------------------------------------------------------------------

void Undersampling::randomUndersampling( const lpmldata::DataPackage& aDataPackage )
{			
	auto majorityKeys   = aDataPackage.getMajorityKeys();
	auto majoritySize   = majorityKeys.size();
	auto sizeDifference = aDataPackage.getMajorityCount() - aDataPackage.getMinorityCount();

	std::random_device rd;
	std::mt19937 rng( rd() );
	std::uniform_int_distribution< int > dice( 0, majoritySize - 1 );

	std::set< int > randomNumberTracker;
	if ( mAuto == true )
	{
		while ( randomNumberTracker.size() != sizeDifference )
		{
			int randomNumber = dice( rng );
			auto key         = majorityKeys.at( randomNumber );

			randomNumberTracker.insert( randomNumber );
			mChoosenSamples.push_back( key );
		}
	}
	else if ( mAuto == false )
	{
		while ( randomNumberTracker.size() != mUndersamplingAmount )
		{
			int randomNumber = dice( rng );
			auto key         = majorityKeys.at( randomNumber );

			randomNumberTracker.insert( randomNumber );
			mChoosenSamples.push_back( key );
		}
	}
}

//-----------------------------------------------------------------------------

void Undersampling::tomekLinks( const lpmldata::DataPackage& aDataPackage )
{
	QMap< QString, QVariantList > minorityFeatures;
	QMap< QString, QVariantList > majorityFeatures;
	QMap< QString, QVariantList > allFeatures;
	QMap< QString, QString > allLabels;

	auto majorityKeys  = aDataPackage.getMajorityKeys();
	auto minorityKeys  = aDataPackage.getMinorityKeys();
	auto minorityLabel = aDataPackage.getMinorityLabel();	

	//Read out the feature values for each sample in minority class from the dataset and store each sample-values pair in a QMap.
	for ( int i = 0; i < minorityKeys.size(); ++i )
	{
		auto key = minorityKeys.at( i );
		auto featureVector = aDataPackage.featureVector( key );
		QString actualLabel = aDataPackage.labelDatabase().valueAt( key, aDataPackage.activeLabelIndex() ).toString();

		minorityFeatures.insert( key, featureVector );
		allFeatures.insert( key, featureVector );
		allLabels.insert( key, actualLabel );
	}

	//Read out the feature values for each sample in majority class from the dataset and store each sample-values pair in a QMap.
	for ( int i = 0; i < majorityKeys.size(); ++i )
	{
		auto key            = majorityKeys.at( i );		
		auto featureVector  = aDataPackage.featureVector( key );
		QString actualLabel = aDataPackage.labelDatabase().valueAt( key, aDataPackage.activeLabelIndex() ).toString();
		
		majorityFeatures.insert( key, featureVector );
		allFeatures.insert( key, featureVector );
		allLabels.insert( key, actualLabel );
	}

	//Find Tomek Links
	for ( int i = 0; i < minorityFeatures.size(); ++i )
	{
		auto minorityFeature = minorityFeatures.values().at( i );
		auto minorityName    = minorityFeatures.keys().at( i );

		for ( int j = 0; j < majorityFeatures.size(); ++j )
		{
			auto majorityFeature  = majorityFeatures.values().at( j );
			auto majorityName     = majorityFeatures.keys().at( j );
			auto distance         = aDataPackage.distance( minorityFeature, majorityFeature );			
			auto allFeatureValues = allFeatures.values();
			auto allFeatureKeys   = allFeatures.keys();
			int counter = 0;

			QVector< int  > counters;
			counters.resize( allFeatures.size() );
			counters.fill( 0 );			

#pragma omp parallel for schedule( dynamic )
			for ( int k = 0; k < allFeatures.size(); ++k )
			{
				auto feature     = allFeatureValues.at( k );
				auto featureName = allFeatureKeys.at( k );
				counters[ k ]    = counterDeterminant( feature, featureName, minorityFeature, minorityName, majorityFeature, majorityName, distance );
			}

			for ( int k = 0; k < counters.size(); ++k )
			{
				counter += counters.at( k );
			}

			if ( counter == allFeatures.size() - 2 )
			{
				//All elements included into TL pairs without repetition
				if ( mChoosenSamples.contains( minorityName ) == false && mChoosenSamples.contains( majorityName ) == false )
				{
					mChoosenSamples.push_back( minorityName );
					mChoosenSamples.push_back( majorityName );
				}
			}
			counter = 0; //reset counter
		}
	}

	//Filter the mChoosenSamples (samples to be removed) from minorities and keep majorities only (minorities shall be cleaned out as well for large data)
	for ( int i = 0; i < mChoosenSamples.size(); ++i )
	{
		auto key = mChoosenSamples.at( i );

		for ( int j = 0; j < allLabels.size(); ++j )
		{
			if ( allLabels.keys().at( j ) == key )
			{
				auto value = allLabels.values().at( j ).toInt();

				if ( value == minorityLabel )
				{
					mChoosenSamples.removeAll( key );
				}
			}
		}
	}
}

//-----------------------------------------------------------------------------


}