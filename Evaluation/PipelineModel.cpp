#include <Evaluation/PipelineModel.h>
#include <Evaluation/RandomForestModel.h>
#include <Evaluation/ConfusionMatrixAnalytics.h>
#include <Evaluation/RandomForestOptimizer.h>
#include <Evaluation/FeatureSelection.h>
#include <Evaluation/Oversampling.h>
#include <Evaluation/IsolationForest.h>
#include <Evaluation/Undersampling.h>
#include <Evaluation/PCA.h>
#include <DataRepresentation/TabularData.h>

namespace dkeval
{
//-----------------------------------------------------------------------------

PipelineModel::PipelineModel( QSettings* aPluginModelSettings, QVector< std::shared_ptr< Node > > aPipeline, lpmldata::DataPackage& aDataPackage )
:
	AbstractModel( aPluginModelSettings ),
	mDPActions(),
	mPipeline( aPipeline ),
	mPluginModel( nullptr ),
	mDataPackage( aDataPackage ),
	mRanges(),
	mFitness( DBL_MAX ),
	mFoldId( 1 )
{
	//create parameter list for each pre-processing algorithm
	for ( auto& algorithm : mPipeline )
	{		
		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "FeatureSelection" )
		{
			//auto maxFeatureCount = (int)std::sqrt( majoritySampleCount * 2 ); //Fulfill the curse of dimensionality requirements to avoid model overfitting
			auto totalFeatureCount = aDataPackage.featureDatabase().headerNames().size(); 
			QVariantList featureCount;		
		
			for( int i = std::min( 3, totalFeatureCount ); i <= totalFeatureCount; ++i )
			{
				featureCount.push_back( i );
			}

			mRanges.insert( algorithm->element + "/featureCount", featureCount );

			QVariantList rankMethod;
			rankMethod.push_back( "RSquared" );
			mRanges.insert( algorithm->element + "/rankMethod", rankMethod );
		}

		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "IsolationForest" )
		{
			QVariantList treeCount;
	
			treeCount.push_back( aDataPackage.featureCount() * 5 );
			treeCount.push_back( aDataPackage.featureCount() * 10 );
			treeCount.push_back( aDataPackage.featureCount() * 20 );

			mRanges.insert( "IsolationForest/treeCount", treeCount );
		}

		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "Oversampling" )
		{
			auto minorityCount = aDataPackage.minorityCount(); //Max number of minority samples within training dataset
			QVariantList neighboursCount;
			for ( int i = 1; i < 10 /*minorityCount*/; ++i )
			{
				neighboursCount.push_back( i );
			}

			mRanges.insert( "Oversampling/neighboursNumber", neighboursCount );



			auto totalSampleCount = aDataPackage.featureDatabase().keys().size();
			QVariantList m_neighboursCount;
			for ( int i = 1; i <= 20 /*totalSampleCount*/; ++i ) //Max number of all samples within training dataset
			{
				m_neighboursCount.push_back( i );
			}

			mRanges.insert( "Oversampling/m_neighboursNumber", m_neighboursCount );

			

			QVariantList n_neighboursCount; //borderline minorty count
			for ( int i = 1; i <= 10 /*minorityCount*/; ++i ) //Test with full minority count
			{
				n_neighboursCount.push_back( i );
			}

			mRanges.insert( "Oversampling/n_neighboursNumber", n_neighboursCount );


			QVariantList oversamplingPercentage;
			for ( int i = 50; i <= 1000; i+=50 ) 
			{
				oversamplingPercentage.push_back( i );
			}

			mRanges.insert( "Oversampling/oversamplingPercentage", oversamplingPercentage );


			QVariantList automatic;
			automatic.push_back( true );
			automatic.push_back( false );

			mRanges.insert( "Oversampling/auto", automatic );


			QVariantList type;
			type.push_back( "SMOTE" );
			type.push_back( "BSMOTE" );
			/*type.push_back( "MWMOTE" );*/
			type.push_back( "RandomOversampling" );
			//type.push_back( "ADASYN" );

			mRanges.insert( "Oversampling/type", type );
		}

		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "Undersampling" )
		{
			QVariantList type;
			type.push_back( "RandomUndersampling" );
			type.push_back( "TomekLink" );			

			mRanges.insert( "Undersampling/type", type );

			//Undersampling is automatically determined within the algorithm based on minority subgroup, not needed as a parameter input
		}

		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "PCA" )
		{
			QVariantList preservationPercentage;
			for ( int i = 90; i < 100; ++i ) 
			{
				preservationPercentage.push_back( i );
			}

			mRanges.insert( "PCA/preservationPercentage", preservationPercentage );
		}		
	}
}

//-----------------------------------------------------------------------------

void PipelineModel::set( const QVector< double >& aParameters )
{	
	QVector< double > normalizedParameters;

	double min = DBL_MAX;
	double max = -DBL_MAX;

	for ( auto& parameter : aParameters )
	{
		if ( parameter < min ) min = parameter;
		if ( parameter > max ) max = parameter;
	}

	for ( auto& parameter : aParameters )
	{
		auto normalizedParameter = ( parameter - min ) / ( max - min );

		if ( normalizedParameter != normalizedParameter )
		{
			/*qDebug() << "normalized parameter is nan";*///parameter is 1.0000
			normalizedParameter = 1.0;
		}

		normalizedParameters.push_back( normalizedParameter );
	}

	auto currentDataPackage     = mDataPackage;	
	auto pipelineSettingsPath   = mSettings->fileName().split( "Settings.ini" ).at( 0 );
	QSettings* pipelineSettings = new QSettings( pipelineSettingsPath + QString::number( mFoldId ) + "pipelineSettings.ini", QSettings::IniFormat );
	
	clearCache();
	
	auto parameterKeys = mRanges.keys();
	for ( int i = 0; i < normalizedParameters.size(); ++i )	
	{
		auto parameterName   = parameterKeys.at( i );
		auto parameterValues = mRanges.value( parameterName );
		int parameterIndex   = ( parameterValues.size() - 1 ) * normalizedParameters.at( i );
		auto parameterValue  = parameterValues.at( parameterIndex );
		pipelineSettings->setValue( parameterName, parameterValue );
	}
	
	pipelineSettings->sync();
	
	//create mDPActions
	for ( auto& algorithm : mPipeline )
	{
		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "FeatureSelection" )
		{			
			std::shared_ptr< FeatureSelection > fs = std::make_shared< FeatureSelection >( pipelineSettings );
			mDPActions.push_back( fs );
		}
		
		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "IsolationForest" )
		{
			std::shared_ptr< IsolationForest > isf = std::make_shared< IsolationForest >( pipelineSettings );
			mDPActions.push_back( isf );
		}

		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "Oversampling" )
		{
			std::shared_ptr< Oversampling > os = std::make_shared< Oversampling >( pipelineSettings );
			mDPActions.push_back( os ); 
		}

		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "Undersampling" )
		{
			std::shared_ptr< Undersampling > us = std::make_shared< Undersampling >( pipelineSettings );
			mDPActions.push_back( us ); 
		}

		//----------------------------------------------------------------------------------------------
		if ( algorithm->element == "PCA" )
		{
			std::shared_ptr< PCA > pca = std::make_shared< PCA >( pipelineSettings );
			mDPActions.push_back( pca ); 
		}
	}
	
	for ( auto dpaction : mDPActions )
	{		
		dpaction->build( currentDataPackage ); 
		currentDataPackage = dpaction->run( currentDataPackage );	
		auto id            = dpaction->id();		
	}	
	

	lpmleval::RandomForestModel* model = new lpmleval::RandomForestModel( mSettings );
	auto analytics                     = new lpmleval::ConfusionMatrixAnalytics( mSettings, &currentDataPackage ); 
	auto optimizer                     = new lpmleval::RandomForestOptimizer( mSettings, &currentDataPackage, model, analytics );
	
	optimizer->build();
	mFitness = analytics->rocDistance();
	
	mPluginModel = model;

	delete analytics;
	delete optimizer;
	delete pipelineSettings;
	
	

}

//-----------------------------------------------------------------------------

QVariant PipelineModel::evaluate( const QVector< double >& aFeatureVector )
{
	return mPluginModel->evaluate( aFeatureVector );
}

//-----------------------------------------------------------------------------

int PipelineModel::inputCount()
{
	return mRanges.size();
}

//-----------------------------------------------------------------------------

PipelineModel::~PipelineModel()
{	
	mRanges.clear();
}

//-----------------------------------------------------------------------------

void PipelineModel::clearCache()
{
	for ( auto element : mDPActions )
	{
		if ( element != nullptr )
		{
			element = nullptr;
		}
	}
	mDPActions.clear();

	if ( mPluginModel != nullptr )
	{
		delete mPluginModel;
		mPluginModel = nullptr;
	}
}

//-----------------------------------------------------------------------------
}