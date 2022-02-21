#include <Evaluation/DataOptimizer.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

QVector< double > DataOptimizer::generateMissingValues( QVector< double >& aFeatureVector ) const
{	
	for ( auto& element : aFeatureVector )
	{
		auto value = QString::number( element );
		if ( value == "NA" )
		{
			double sum = 0;
			std::for_each( aFeatureVector.begin(), aFeatureVector.end(), 
				           [ &sum ]( double x ) { sum += x; } );

			auto denominator = aFeatureVector.size();
			double average   = sum / denominator;
			element          = average;
		}		
	}	
	
	return aFeatureVector;
}

//-----------------------------------------------------------------------------

void dkeval::DataOptimizer::build()
{
	auto header = mFDB.headerNames();

	for ( auto& featureName : header )
	{
		QVector< double > featureVector;
		auto featureVariantList = mFDB.column( featureName );
		
		for ( auto& variant : featureVariantList )
		{
			featureVector.push_back( variant.toDouble() );
		}

		if ( isRedundand( featureVector ) )
		{
			mRedundandFeatures << featureName;
		}
		else if ( hasEnoughTrueValues( featureVector ) )
		{
			generateMissingValues( featureVector );
		}
	}

	auto purifiedHeader = header.toSet().subtract( mRedundandFeatures.toSet() ).toList();

	// Create new dataset
	lpmldata::TabularData updatedFDB = featureDatabaseSubset( purifiedHeader, mFDB );

	mFDB = updatedFDB;
}
//-----------------------------------------------------------------------------

bool DataOptimizer::hasEnoughTrueValues( const QVector< double >& aFeatureVector ) const
{
	auto falseEntryNumber = 0;

	for ( auto& element : aFeatureVector )
	{
		auto value = QString::number( element );
		if ( value == "NA" )
		{
			falseEntryNumber++;
		}
	}

	double totalNumberOfElements = aFeatureVector.size();
	auto falseEntryPercentage    = ( double( falseEntryNumber ) / totalNumberOfElements ) * 100;

	if ( falseEntryPercentage < 0.2 )
	{
		return true;
	}
	else
	{
		return false;
	}
}

//-----------------------------------------------------------------------------

bool DataOptimizer::isRedundand( const QVector< double >& aFeatureVector ) const
{
	auto vectorSize = aFeatureVector.size();
	double sum      = 0;

	std::for_each( aFeatureVector.begin(), aFeatureVector.end(), [ &sum ]( double x ) { sum += x; } );

	if ( sum == 0 || sum == vectorSize || !hasEnoughTrueValues( aFeatureVector ) )
	{
		return true;
	}
	else
	{
		return false;
	}
}

//-----------------------------------------------------------------------------

lpmldata::TabularData DataOptimizer::featureDatabaseSubset( const QStringList& aPurifiedHeader, const lpmldata::TabularData& aFDB ) const
{
	lpmldata::TabularData filteredFDB;
	filteredFDB.setHeader( aPurifiedHeader );

	auto originalFeatureNames = aFDB.headerNames();
	auto keys = aFDB.keys();

	QVector< int > indices;

	for ( auto& featureName : aPurifiedHeader )
	{
		for ( int i = 0; i < originalFeatureNames.size(); ++i )
		{
			auto originalName = originalFeatureNames.at( i );
			auto featureIndex = -1;


			if ( featureName == originalName )
			{
				featureIndex = i;
				indices.push_back( featureIndex );
			}
		}
	}

	for ( auto& key : keys )
	{
		QVariantList filteredFeatureVector;

		for ( int i = 0; i < indices.size(); ++i )
		{
			filteredFeatureVector.push_back( aFDB.valueAt( key, indices.at( i ) ) );
		}

		filteredFDB.insert( key, filteredFeatureVector );
	}

	return filteredFDB;
}

//-----------------------------------------------------------------------------

}

