#include <Evaluation/AbstractMLAgent.h>
#include <Evaluation/MLAgentFactory.h>
#include <FileIo/TabularDataFileIo.h>
#include <QDebug>


namespace lpmleval
{

//-----------------------------------------------------------------------------

AbstractMLAgent::AbstractMLAgent( QSettings* aSettings )
:
	mSettings( aSettings ),
	mDataPackage( nullptr ),
	mOptimizer( nullptr ),
	mModel( nullptr ),
	mAnalytics( nullptr ),
	mMLUID( "NA" ),
	mComplexity( "NA" ),
	mDataPackageType( "NA" ),
	mAnalyticsType( "NA" ),
	mModelType( "NA" ),
	mOptimizerType( "NA" ),
	mInputMLAgents()
{

	mSettings->beginGroup( "Global" );
	mMLUID = mSettings->value( "MLAgent/MLUID" ).toString();
	mComplexity = mSettings->value( "MLAgent/Complexity" ).toString();
	mSettings->endGroup();

	mSettings->beginGroup( "DataPackage" );
	mDataPackageType = mSettings->value( "Type" ).toString();
	mSettings->endGroup();

	mSettings->beginGroup( "Analytics" );
	mAnalyticsType = mSettings->value( "Type" ).toString();
	mSettings->endGroup();

	mSettings->beginGroup( "Model" );
	mModelType = mSettings->value( "Type" ).toString();
	mSettings->endGroup();

	mSettings->beginGroup( "Optimizer" );
	mOptimizerType = mSettings->value( "Type" ).toString();
	mSettings->endGroup();

	if ( mComplexity == "Complex" )
	{
		generateMLAgents();  // Load up the input MLAgents.
	}
}

//-----------------------------------------------------------------------------

AbstractMLAgent::~AbstractMLAgent()
{
}

//-----------------------------------------------------------------------------

void AbstractMLAgent::train()
{
	if ( mOptimizer != nullptr )
	{
		//qDebug() << "MLAgent" << mMLUID << "Training started.";
		mOptimizer->build();
		mModel->set( mOptimizer->result() );  // Is this really necessary??? The build function above does the job already.

		//qDebug() << "MLAgent" << mMLUID << "Training finished.";
	}
	else
	{
		qDebug() << "MLAgent" << mMLUID << "The optimizer object is a nullptr!";
	}
}

//-----------------------------------------------------------------------------

void AbstractMLAgent::validate( lpmldata::DataPackage* aDataPackage )
{
	double result = -DBL_MAX;

	if ( mComplexity == "Atomic" )
	{
		mAnalytics->setDataPackage( aDataPackage );
		if ( mAnalytics != nullptr && mOptimizer != nullptr )
		{
			if ( mModel != nullptr )
			{
				result = mAnalytics->evaluate( mModel );
				qDebug() << "MLAgent" << mMLUID << "Validation result: " << result << mAnalytics->unit();
			}
			else
			{
				qDebug() << "MLAgent" << mMLUID << "The model object is a nullptr!";
			}
		}
	}
	else if ( mComplexity == "Complex" )
	{
		// TODO: Take the aDataPackage feature vectors and give them to mInputMLJobs. 
		// TODO: Take the return value of mInputMLJobs and put them to a meta feature vector.
		// TODO: Add the generated meta feature vector to a metatable. Once done with all original feature vectors, create a new MetaDataPackage
		// TODO: Evaluate the mModel with mAnalytics over the newly created MetaDataPackage which contains the metatable.

		qDebug() << "MLAgent" << mMLUID << "Valdiation result: " << result << mAnalytics->unit();
		qDebug() << "MLAgent" << mMLUID << "Complex validation branch implementation missing.";
	}



	qDebug() << "MLJob" << mMLUID << "validation finished.";
	// TODO: Plot out the results to the screen.
}

//-----------------------------------------------------------------------------

const QVariant AbstractMLAgent::evaluate( QVector< double > aFeatureVector )
{
	if ( mModel != nullptr )
	{
		return mModel->evaluate( aFeatureVector );
	}
	else
	{
		qDebug() << "MLAgent" << mMLUID << "The model object is a nullptr!";
		return "NA";
	}	
}

//-----------------------------------------------------------------------------

void AbstractMLAgent::generateMLAgents()
{
	mSettings->beginGroup( "Global" );
	QString inputMLAgentString = mSettings->value( "MLAgent/InputMLJobs" ).toString();
	mSettings->endGroup();

	QStringList inputMLAgents = inputMLAgentString.split( "," );
	mInputMLAgents.clear();
	lpmleval::MLAgentFactory factory;

	for ( auto inputMLAgentPath : inputMLAgents )
	{ 
		mInputMLAgents.push_back( factory.generate( inputMLAgentPath ) );  // TODO: Here, generate MLAgents not with default settings but with already trained models! Use the load function.
	}
}

//-----------------------------------------------------------------------------
//
//lpmldata::DataPackage* AbstractMLAgent::buildMetaTable( lpmldata::DataPackage* aDataPackage )
//{
//	// Read out the FDB, LDB and labelName values from aDataPackage.
//	//auto FDB = aDataPackage->featureDB();
//	auto FDB = aDataPackage->featureDatabase();
//	auto LDB = aDataPackage->labelDatabase();
//	auto labelName = aDataPackage->labelName(); //Check
//
//	lpmldata::TabularData metaTable;
//
//	// Generate the meta table header.
//	QStringList metaTableHeaderNames;
//	for ( auto MLAgent : mInputMLAgents )
//	{
//		metaTableHeaderNames.push_back( MLAgent->uid() );
//	}
//	metaTable.setHeader( metaTableHeaderNames );
//
//	// Evaluate the FDB records by the mInputMLJobs to generate a new metatable.
//	for ( auto key : FDB.keys() )
//	{
//		auto featureVectorVariant = FDB.value( key );
//		
//		// Build the input feature vector.
//		QVector< double > featureVector;
//		for ( int i = 0; i < featureVector.size(); ++i )
//		{
//			featureVector.push_back( featureVectorVariant.at( i ).toDouble() );
//		}
//
//		// Build the generated meta feature vector.
//		QVariantList metatableFeatuerVector;
//		for ( auto MLAgent : mInputMLAgents )
//		{
//			metatableFeatuerVector.push_back( MLAgent->evaluate( featureVector ) );
//		}
//
//		// Save the new feature vector to the metatable.
//		metaTable.insert( key, metatableFeatuerVector );
//	}
//	
//	return new lpmldata::DataPackage( metaTable, LDB, labelName );
//}

//-----------------------------------------------------------------------------

}
