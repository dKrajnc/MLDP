#include <IsolationForest.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

void IsolationForest::build( const lpmldata::DataPackage& aDataPackage )
{
	auto FDB          = aDataPackage.featureDatabase();
	auto featureNames = FDB.headerNames();
	auto keys         = aDataPackage.commonKeys();
	
	

	QStringList names; //sample keys
	QVector< QVector< int > >pathLenghts;
	QMap< QString, double > averagePathLenghts;

	for ( int index = 0; index < mTreesnumber; index++ ) 
	{
		//Randomly select a feature from the dataset
		auto headerSize = featureNames.size();

		QVector< double > featureVector;//Randomly selected feature vector				

		std::random_device rd;
		std::mt19937 rng( rd() );
		std::uniform_int_distribution< int > dice( 0, headerSize - 1 ); 

		int featureIndex = dice( rng );

		for ( int i = 0; i < 1; ++i )
		{
			for ( int j = 0; j < keys.size(); ++j )
			{
				auto key   = keys.at( j );
				auto value = FDB.valueAt( key, featureIndex ).toDouble();
				
				featureVector.push_back( value );
				names.push_back( key );
			}
		}		
		//Populate the tree	
		mRoot = createNode( featureVector );		

		addNode( mRoot );

		
		//Calculate the path lenght of a sample in the tree
		QVector< int > lenghts;

		for ( int k = 0; k < featureVector.size(); ++k )
		{			
			auto path = pathLenght( featureVector.at( k ), mRoot );		
			auto name = names.at( k );
			
			lenghts.push_back( path );
			resetCounter();			
		}
		pathLenghts.push_back( lenghts );
	}

	//Calculate the average path lenght for each sample
	for ( int i = 0; i < pathLenghts.at( 0 ).size(); ++i )
	{
		QVector< int > temp;

		for ( int j = 0; j < pathLenghts.size(); ++j )
		{
			auto value = pathLenghts.at( j ).at( i );

			temp.push_back( value );
		}
		auto value  = averagePathLenght( temp );
		auto name   = names.at( i );

		averagePathLenghts.insertMulti( name, value );
	}
	
	//Calculate the average path lenght of all average path lenghts
	auto avgPath = averagePath( averagePathLenghts );		


	


	for ( int i = 0; i < averagePathLenghts.size(); ++i )
	{
		auto anomaly = anomalyScore( avgPath, averagePathLenghts.values().at( i ) );

		if ( anomaly >= 0.6 )
		{
			mOutliers.push_back( averagePathLenghts.keys().at( i ) );
		}
	}	
}

//-----------------------------------------------------------------------------

lpmldata::DataPackage IsolationForest::run( const lpmldata::DataPackage& aDataPackage )
{
	auto keys         = aDataPackage.commonKeys();
	auto purifiedKeys = keys.toSet().subtract( mOutliers.toSet() ).toList();

	auto updatedFDB = aDataPackage.sampleDatabaseSubset( purifiedKeys );
	auto updatedLDB = aDataPackage.labelDatabaseSubset( purifiedKeys );	

	lpmldata::DataPackage result( updatedFDB, updatedLDB );
	return result;
}

//-----------------------------------------------------------------------------

std::shared_ptr< IsolationForest::Node > IsolationForest::createNode( QVector< double >& aValue )
{
	std::shared_ptr< Node > node = std::make_shared< Node >();
	node->value = aValue;
	node->left  = nullptr;
	node->right = nullptr;

	return node;
}

//-----------------------------------------------------------------------------

void IsolationForest::addNode( std::shared_ptr< Node > aPointer )
{
	if ( mRoot == nullptr )
	{
		qDebug() << "Error - the root is empty!";
	}
	else 
	{		
		//Determine min and max values for the selected feature 
		QVector< double > temp = aPointer->value;

		std::sort( temp.begin(), temp.end() );
		
		auto min = temp.first();
		auto max = temp.last();


		//generate a random value between min and max
		std::random_device rd;
		std::mt19937 rng( rd() );
		std::uniform_real_distribution< double > distribute( min, max );

		auto splitValue = distribute( rng );

		QVector< double > lower;
		QVector< double > higher;

		for ( int i = 0; i < aPointer->value.size(); ++i )
		{
			if ( aPointer->value.at( i ) < splitValue )
			{
				lower.push_back( aPointer->value.at( i ) );
			}
			else
			{
				higher.push_back( aPointer->value.at( i ) );
			}
		}

		bool brakeTest = false;

		if ( lower.isEmpty() == true || higher.isEmpty() == true )
		{	
			brakeTest = true;
		}

		if ( brakeTest != true )
		{
			if ( lower.size() > 1 )
			{
				aPointer->left = createNode( lower );
				addNode( aPointer->left );
			}
			else if ( lower.size() == 1 )
			{
				aPointer->left = createNode( lower );
			}

			if ( higher.size() > 1 )
			{
				aPointer->right = createNode( higher );
				addNode( aPointer->right );
			}
			else if ( higher.size() == 1 )
			{
				aPointer->right = createNode( higher );
			}
		}
	
	}	
}

//-----------------------------------------------------------------------------

void IsolationForest::removeNode( std::shared_ptr< Node > aPointer )
{
	if ( aPointer != nullptr )
	{
		if ( aPointer->left != nullptr )
		{
			removeNode( aPointer->left );
		}
		if ( aPointer->right != nullptr )
		{
			removeNode( aPointer->right );
		}		
		
		aPointer = nullptr;
	}
	else
	{
		qDebug() << "mRoot is empty!";
	}
}

//-----------------------------------------------------------------------------

int IsolationForest::pathLenght( double aValue, std::shared_ptr< Node > aRoot )
{
	if ( aRoot->value.contains( aValue ) == true )
	{						
		mCounter++;
	}		
	
	if ( aRoot->left != nullptr && aRoot->value.size() > 1 )
	{
		pathLenght( aValue, aRoot->left );
	}

	if ( aRoot->right != nullptr && aRoot->value.size() > 1 )
	{
		pathLenght( aValue, aRoot->right );
	}

	return mCounter;
}

//-----------------------------------------------------------------------------

void IsolationForest::resetCounter()
{
	mCounter = 0;
}

//-----------------------------------------------------------------------------

double IsolationForest::averagePathLenght( QVector< int >& aPathLenghts )
{
	auto denominator = aPathLenghts.size();
	double sum       = 0;
	double average   = 0;

	std::for_each( aPathLenghts.begin(), aPathLenghts.end(),
				   [ &sum ]( auto x ) { sum += x; } );

	average = sum / denominator;

	return average;
}

//-----------------------------------------------------------------------------

double IsolationForest::averagePath( QMap< QString, double >& aPathLenghts )
{
	auto denominator = aPathLenghts.values().size();
	double sum     = 0;
	double average = 0;

	for ( auto& element : aPathLenghts.values() )
	{
		sum += element;
	}

	average = sum / denominator;

	return average;
}
//-----------------------------------------------------------------------------

double IsolationForest::unsuccessfulSearchLenght( int aInstancesCount )
{
	const double eulerConstant = 0.5772156649;
	const double Ln_i          = 1.57079633;

	auto reducedInstanceCount  = aInstancesCount - 1;
	auto harmonic              = Ln_i + eulerConstant;

	auto left   = 2 * harmonic * ( reducedInstanceCount );
	auto right  = ( 2 * reducedInstanceCount ) / aInstancesCount;

	auto result = left - right;
	
	return result;
}

//-----------------------------------------------------------------------------

double IsolationForest::anomalyScore( double aUnsuccessfulSearchLenght, double aAveragePathLenght )
{
	auto divided = -( aAveragePathLenght / aUnsuccessfulSearchLenght) ;

	auto anomaly = std::pow( 2, divided );

	return anomaly;
}


//-----------------------------------------------------------------------------

IsolationForest::~IsolationForest()
{
		removeNode( mRoot );	
}

//-----------------------------------------------------------------------------

}