#include <Evaluation/Oversampling.h>
#include <Evaluation/IsolationForest.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

void Oversampling::build( const lpmldata::DataPackage& aDataPackage )
{
	mDataPackage = &aDataPackage;
	mLabel       = mDataPackage->getMinorityLabel();
		
	//calculate the difference between samples
	mSamplesDifference = mDataPackage->getMajorityCount() - mDataPackage->getMinorityCount();
	

	//Switch for method choice
	if ( mMethod == "SMOTE" )
	{
		smote();
	}
	else if ( mMethod == "BSMOTE" )
	{
		bSmote();
	}
	else if ( mMethod == "RandomOversampling" )
	{
		randomOVersampling();
	}
	else
	{
		qDebug() << "Error - no valid method selected!\n";
	}
			
}

//-----------------------------------------------------------------------------

QMap< double, QString > Oversampling::getNeighbours( const QString& aEvaluatedKey, const QStringList& aNeighbourKeys )
{
	QMap< double, QString > neighbours;

	auto evaluatedKeyFeatureVector = mDataPackage->featureVector( aEvaluatedKey );

	
	for each ( auto& neighbour in aNeighbourKeys )
	{
		if ( aEvaluatedKey != neighbour )
		{
			auto neighbourFeatureVector = mDataPackage->featureVector( neighbour );
			auto distance               = mDataPackage->distance( evaluatedKeyFeatureVector, neighbourFeatureVector );
		
			neighbours.insertMulti( distance, neighbour );
		}
	}
	return neighbours;
}

//-----------------------------------------------------------------------------

QMap< double, QString > Oversampling::getNearestNeighbours( const QString& aEvaluatedKey, const QStringList& aNeighbourKeys, const int& aNearestNeighboursCount )
{
	auto neighbours = getNeighbours( aEvaluatedKey, aNeighbourKeys );
	
	QMap< double, QString > kNN;
	for ( int i = 0; i < neighbours.size(); ++i )
	{
		if ( i == aNearestNeighboursCount )
		{
			break;
		}
		else
		{
			kNN.insertMulti( neighbours.keys().at( i ), neighbours.values().at( i ) );
		}
		
	}

	return kNN;
}


//-----------------------------------------------------------------------------

void Oversampling::filterNearestNeighbours( QMap< double, QString >& aNeighbours, const int& aNeighboursNumber ) //Filter aNeighbours map - pass by &
{
	auto counter = 0; 

	for ( auto iterator = aNeighbours.begin(); iterator != aNeighbours.end(); )
	{		
		if ( counter >= aNeighboursNumber )
		{
			aNeighbours.erase( iterator++ );				
		}
		else
		{
			iterator++;
		}
		counter++;		
	}
}

//-----------------------------------------------------------------------------

QMap< QString, QVector< double > > Oversampling::generateSyntheticSample( const QString& aSampleKey, const double& aDistance )
{
	QMap< QString, QVector< double > > synthticSample;
	QVector< double > syntheticValues;	
	
	
	std::uniform_real_distribution< double > dDice( 0.0, 1.0 );
	std::uniform_int_distribution< int > iDice( 1, 1000000 );
	auto randomDouble  = dDice( *mRng );
	auto randomInt     = iDice( *mRng );


	QString syntheticName = "Synthetic sample " + QString::number( randomInt );
	mSyntheticNames << syntheticName;

	double product           = randomDouble * aDistance;	
	auto sampleFeatureVector = mDataPackage->featureVector( aSampleKey );		


	for ( auto& value : sampleFeatureVector )
	{
		syntheticValues.push_back( value.toDouble() + product );
	}	
	synthticSample.insert( syntheticName, syntheticValues );
	
	return synthticSample;
}

//-----------------------------------------------------------------------------

QStringList Oversampling::borderlineMajorities( const QStringList& aMinorityKeys, const QStringList& aMajorityKeys )
{
	QStringList borderlineMajorities;

	for ( auto& element : aMinorityKeys )
	{
		auto neighbours = getNeighbours( element, aMajorityKeys );

		filterNearestNeighbours( neighbours, mM_NeighboursNumber ); //mM_NeighboursNumber == mNN, k2

		for ( auto& name : neighbours.values() )
		{
			borderlineMajorities << name;
		}
	}

	borderlineMajorities.removeDuplicates();

	return borderlineMajorities;
}

//-----------------------------------------------------------------------------

QStringList Oversampling::borderlineMinorities( const QStringList& aMinorityKeys, const QStringList& aMajorityKeys )
{
	QStringList borderlineMinorities;

	for ( auto& element : aMajorityKeys )
	{
		auto neighbours = getNeighbours( element, aMinorityKeys );

		filterNearestNeighbours( neighbours, mN_NeighboursNumber ); //mN_NeighboursNumber == k3  

		for ( auto& name : neighbours.values() )
		{
			borderlineMinorities << name;
		}
	}

	borderlineMinorities.removeDuplicates();

	return borderlineMinorities;
}

//-----------------------------------------------------------------------------

double Oversampling::closenessFactor( const QString& aMajorityKey, const QString& aMinorityKey )
{
	const int cFactor = 100; //Suggested by the literature
	const int cMax    = 2; //Suggested by the literature
	
	auto majorityFeatureVector = mDataPackage->featureVector( aMajorityKey );
	auto minorityFeatureVector = mDataPackage->featureVector( aMinorityKey );
	auto distance              = mDataPackage->distance( majorityFeatureVector, minorityFeatureVector );
	auto vectorLenght          = majorityFeatureVector.size();

	auto normalized = distance / vectorLenght;	
	auto inverse    = 1 / normalized;

	if ( inverse > cFactor ) inverse = cFactor;		

	return ( inverse / cFactor ) * cMax;
}

//-----------------------------------------------------------------------------

double Oversampling::densityFactor( const double& aClosenessFactor, const QString& aMajorityKey, const QStringList& aBorderlineMinorities )
{
	auto closeFactorSum = 0;
	
	for ( auto& borderlineMinority : aBorderlineMinorities )
	{
		auto closenessFactor = this->closenessFactor( aMajorityKey, borderlineMinority );
		closeFactorSum += closenessFactor;
	}
	
	return aClosenessFactor / closeFactorSum;
}

//-----------------------------------------------------------------------------

double Oversampling::informationWeight( const QString& aMajorityKey, const QString& aMinorityKey, const QStringList& aBorderlineMinorities )
{
	auto closenessFactor = this->closenessFactor( aMajorityKey, aMinorityKey );
	auto densityFactor   = this->densityFactor( closenessFactor, aMajorityKey, aBorderlineMinorities );

	return closenessFactor * densityFactor;
}

//-----------------------------------------------------------------------------

double Oversampling::selectionWeight( const QString& aMinorityKey, const QStringList& aBorderlineMinorities, const QStringList& aBorderlineMajorities )
{
	double selectionWeight = 0.0;

	for ( auto& borderlineMajority : aBorderlineMajorities )
	{
		auto informationWeight = this->informationWeight( borderlineMajority, aMinorityKey, aBorderlineMinorities );
		selectionWeight += informationWeight;
	}
	
	return selectionWeight;
}

//-----------------------------------------------------------------------------

QMap< double, QString > Oversampling::selectionProbabilities( QMap< double, QString >& aSelectionWeights )
{
	QMap< double, QString > probabilities;
	double sum = 0;

	std::for_each( aSelectionWeights.keyBegin(), aSelectionWeights.keyEnd(),
		[ &sum ]( auto x ) { sum += x; } );


	for ( auto& weight : aSelectionWeights.keys() )
	{
		auto element = aSelectionWeights.value( weight );
		auto result  = weight / sum;

		probabilities.insertMulti( result, element );
	}
	
	return probabilities;
}

//-----------------------------------------------------------------------------

double Oversampling::averageMinimalDistance( const QStringList& aMinorityKeys )
{
	auto minimalDistanceSum = 0.0;

	for ( auto& firstKey : aMinorityKeys )
	{
		QVector< double > minimalDistances;

		auto firstFeatureVector = mDataPackage->featureVector( firstKey );

		for ( auto& secondKey : aMinorityKeys )
		{
			if ( firstKey != secondKey )
			{
				auto secondFeatureVector = mDataPackage->featureVector( secondKey );
				auto distance            = mDataPackage->distance( firstFeatureVector, secondFeatureVector );

				minimalDistances.push_back( distance );
			}			
		}
		std::sort( minimalDistances.begin(), minimalDistances.end() );

		minimalDistanceSum += minimalDistances.first();
	}
	
	return 1 / minimalDistanceSum;
}

//-----------------------------------------------------------------------------

double Oversampling::averageDistance( const QStringList& aMinorityKeys )
{
	QVector< double > distances;
	double sum = 0.0;

	for ( auto& minorityKey : aMinorityKeys )
	{
		auto neighbours = getNeighbours( minorityKey, aMinorityKeys );

		for ( auto& value : neighbours.keys() )
		{
			distances.push_back( value );
		}
	}

	std::for_each( distances.begin(), distances.end(),
		 [ &sum ]( double x ) { sum += x; } );

	auto average = sum / distances.size();

	return average;
}

//-----------------------------------------------------------------------------

double Oversampling::averageClusterDistance( const QStringList& aFirstCluster, const QStringList& aSecondCluster )
{
	QVector< double > distances;
	double sum = 0.0;

	for ( auto& firstElement : aFirstCluster )
	{
		auto clusterDistances = getNeighbours( firstElement, aSecondCluster );

		for ( auto& value : clusterDistances.keys() )
		{
			distances.push_back( value );
		}
	}

	std::for_each( distances.begin(), distances.end(),
		[ &sum ]( double x ) { sum += x; } );

	auto average = sum / distances.size();

	return average;
}

//-----------------------------------------------------------------------------

double Oversampling::setThreshold( const double& aAverageDistance )
{
	const double C_parameter = 3.0; //suggested by the literature

	return aAverageDistance * C_parameter;
}

//-----------------------------------------------------------------------------

QVector< QStringList > Oversampling::generateClusters( const QStringList& aMinorityKeys )
{
	QVector< QStringList > intialClusters;

	for ( auto& minorityKey : aMinorityKeys )
	{					
		QStringList cluster;
		cluster << minorityKey;
		intialClusters.push_back( cluster );
	}

	findClusters( intialClusters );

	for ( int i = 0; i < intialClusters.size(); ++i )
	{
		if ( intialClusters.at( i ).size() < 2 )
		{
			intialClusters.removeAt( i );
			--i;
		}
	}

	return intialClusters;
}

//-----------------------------------------------------------------------------

QVector< double > Oversampling::toQVectorDouble( const QVariantList& aQVatiantList )
{
	QVector< double > vector;

	for ( auto& qVariant : aQVatiantList )
	{
		vector.push_back( qVariant.toDouble() );
	}

	return vector;
}

//-----------------------------------------------------------------------------

void Oversampling::findClusters( QVector< QStringList >& aInitialClusters )
{	
	QMap< double, QPair< int, int > > distances;
	
	for ( int i = 0; i < aInitialClusters.size(); ++i )
	{		
		auto firstCluster = aInitialClusters.at( i );

		for ( int j = 0; j < aInitialClusters.size(); ++j )
		{
			auto secondCluster = aInitialClusters.at( j );

			if ( i < j ) 
			{
				QPair< int, int > indexes;

				indexes.first        = i;
				indexes.second       = j;
				auto clusterDistance = averageClusterDistance( firstCluster, secondCluster );

				distances.insertMulti( clusterDistance, indexes );
			}
		}
	}

	auto smallestClusterDistance = distances.keys().at( 0 );

	if ( smallestClusterDistance <= mSelectionThreshold )
	{
		auto firstIndex  = distances.values().at( 0 ).first;
		auto secondIndex = distances.values().at( 0 ).second;
		auto firstList   = aInitialClusters.at( firstIndex );
		auto secondList  = aInitialClusters.at( secondIndex );

		firstList.append( secondList );
		aInitialClusters.replace( firstIndex, firstList );
		aInitialClusters.removeAt( secondIndex );

		findClusters( aInitialClusters );
	}		
}


//-----------------------------------------------------------------------------

void Oversampling::smote()
{	
	auto minorityKeys     = mDataPackage->getMinorityKeys();
	int oversampplingRate = mOversamplingAmount / 100;	
	
	if ( mAutomatic == true )
	{
		int counter = 0;

		std::uniform_int_distribution< int > iDice( 0, minorityKeys.size() - 1 );

		while ( counter < mSamplesDifference )
		{
			auto randomInteger = iDice( *mRng );
			auto element       = minorityKeys.at( randomInteger );
			auto neighbours    = getNeighbours( element, minorityKeys );
			filterNearestNeighbours( neighbours, mNeighboursNumber );


			std::uniform_int_distribution< int > iDice( 0, mNeighboursNumber - 1 );
			randomInteger                  = iDice( *mRng );
			auto choosenNeighbourDisstance = neighbours.keys().at( randomInteger );
			mSyntheticSamples.unite( generateSyntheticSample( element, choosenNeighbourDisstance ) );

			counter++;
		}
	}
	else
	{
		for each ( auto& element in minorityKeys )
		{
			auto neighbours = getNeighbours( element, minorityKeys );
			filterNearestNeighbours( neighbours, mNeighboursNumber );

			for ( unsigned int index = 0; index < oversampplingRate; ++index )
			{
				std::uniform_int_distribution< int > iDice( 0, mNeighboursNumber - 1 );
				auto randomInteger            = iDice( *mRng );
				auto choosenNeighbourDistance = neighbours.keys().at( randomInteger );
				mSyntheticSamples.unite( generateSyntheticSample( element, choosenNeighbourDistance ) );
			}
		}
	}	
}

//-----------------------------------------------------------------------------

void Oversampling::bSmote()
{
	QStringList danger;
	int oversampplingRate = mOversamplingAmount / 100;
	auto minorityKeys     = mDataPackage->getMinorityKeys();
	auto majorityKeys     = mDataPackage->getMajorityKeys();
	auto allKeys          = mDataPackage->featureDatabase().keys();

	for each ( auto& element in minorityKeys )
	{
		int majorityNeighboursCount = 0;
		auto neighbours             = getNeighbours( element, allKeys );		
		filterNearestNeighbours( neighbours, mM_NeighboursNumber ); //mM_NeighboursNumber == mNN

		for ( auto& name : neighbours.values() )
		{
			if ( majorityKeys.contains( name ) == true )
			{
				majorityNeighboursCount++;
			}
		}

		int totalNeighboursCount = neighbours.size();	

		if (  ( totalNeighboursCount / 2 <= majorityNeighboursCount )  &&  ( majorityNeighboursCount < totalNeighboursCount ) )
		{
			danger << element;
		}
	}

	if ( !danger.isEmpty() )
	{
		for each( auto& element in danger )
		{
			auto neighbours = getNeighbours( element, minorityKeys );
			filterNearestNeighbours( neighbours, mNeighboursNumber ); //mNeighboursNumber == kNN


			for ( int index = 0; index < oversampplingRate; ++index )
			{
				std::uniform_int_distribution< int > iDice( 0, mNeighboursNumber - 1 );
				auto randomInteger            = iDice( *mRng );
				auto choosenNeighbourDistance = neighbours.keys().at( randomInteger );
				mSyntheticSamples.unite( generateSyntheticSample( element, choosenNeighbourDistance ) );
			}
		}
	}
	else
	{
		//Fill original values
		auto keys = mDataPackage->featureDatabase().keys();		
		
		for ( auto& key : keys )
		{
			QVector< double > featureVector;
			auto vector = mDataPackage->featureDatabase().value( key );

			for ( auto& value : vector )
			{
				featureVector.push_back( value.toDouble() );
			}

			mSyntheticSamples.insert( key, featureVector );
		}
	}
	
}

//-----------------------------------------------------------------------------

void Oversampling::randomOVersampling()
{
	auto FDB              = mDataPackage->featureDatabase();
	auto minorityKeys     = mDataPackage->getMinorityKeys();
	auto majorityKeys     = mDataPackage->getMajorityKeys();
	auto oversamplingRate = majorityKeys.size() - minorityKeys.size();

	QMap< QString, QVariantList > minorityFeatures;

	for ( auto& key : minorityKeys )
	{
		auto feature = FDB.value( key );	

		minorityFeatures.insert( key, feature );
	}
		
	std::uniform_int_distribution< int > dice( 0, minorityKeys.size() - 1 );

	QStringList choosenSamples;

	for ( int i = 0; i < oversamplingRate; ++i )
	{
		int randomNumber = dice( *mRng );
		auto key         = minorityKeys.at( randomNumber );

		choosenSamples.push_back( key );
	}

	for ( int i = 0; i < choosenSamples.size(); ++i )
	{
		auto key = choosenSamples.at( i );

		for ( int j = 0; j < minorityFeatures.size(); ++j )
		{
			auto name          = minorityFeatures.keys().at( j );
			auto featureList   = minorityFeatures.values().at( j );
			auto featureVector = toQVectorDouble( featureList );

			if ( key == name )
			{
				int random        = dice( *mRng );
				QString synthName = "Synthetic sample no:" + QString::number( i ) + QString::number( j ) + QString::number( random );

				mSyntheticSamples.insert( synthName, featureVector );
				mSyntheticNames << synthName;
				continue;
			}
		}
	}
}

//-----------------------------------------------------------------------------

lpmldata::DataPackage Oversampling::run( const lpmldata::DataPackage& aDataPackage )
{
	//Create new dataset
	lpmldata::TabularData updatedFDB;
	lpmldata::TabularData updatedLDB;
		
	updatedFDB = aDataPackage.sampleDatabaseSubset( mSyntheticSamples );
	updatedLDB = aDataPackage.labelDatabaseSubset( mSyntheticNames, mLabel );	

	lpmldata::DataPackage result( updatedFDB, updatedLDB );

	return result;
}

//-----------------------------------------------------------------------------

}