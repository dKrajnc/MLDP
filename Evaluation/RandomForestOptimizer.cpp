/*!
* \file
* Member function definitions for RandomForest class.
*
* \remarks
*
* \authors
* cspielvogel
*/

#include <Evaluation/AbstractOptimizer.h>
#include <Evaluation/RandomForestOptimizer.h>
#include <Evaluation/RandomForestModel.h>
#include <Evaluation/DecisionTreeOptimizer.h>
#include <FileIo/TabularDataFileIo.h>
#include <Evaluation/TabularDataFilter.h>
#include <Evaluation/KernelDensityExtractor.h>
#include <QTime>
#include <math.h>
#include <QDebug>
#include <QMap>
#include <QVector>
#include <QList>
#include <QPair>
#include <QVariant>

namespace lpmleval
{

//-----------------------------------------------------------------------------

RandomForestOptimizer::RandomForestOptimizer( QSettings* aSettings, lpmldata::DataPackage* aDataPackage, lpmleval::AbstractModel* aModel, lpmleval::AbstractAnalytics* aAnalytics )
:
	AbstractOptimizer( aSettings ),
	mDataPackage( aDataPackage ),
	mAnalytics( aAnalytics ),
	mNumberOfTrees( 0 ),					
	mQualityMetric(),						
	mMaxDepth( 0 ),							
	mMinSamplesAtLeaf( 0 ),					
	mKDEAttributesPerSplit( 0 ),			
	mTreeSelection(),							
	mNumberSelectedTrees( 0 ),					
	mBaggingMethod(),							
	mBagFraction( 0.0 ),						
	mBoosting(),								
	mTreeWeights(),								
	mKeysToWeights(),							
	mBoostMultiplier()							
{
	mModel = aModel;

	// Default values for string based parameters
	mQualityMetric = "gain";
	mTreeSelection = "none";
	mBaggingMethod = "normal";
	mBoosting = "none";

	bool isValidmNumberOfTrees;
	bool isValidmMaxDepth;
	bool isValidmMinSamplesAtLeaf;
	bool isValidmKDEAttributesPerSplit;
	bool isValidmNumberSelectedTrees;
	bool isValidmBagFraction;

	mNumberOfTrees = aSettings->value( "Optimizer/NumberOfTrees" ).toInt( &isValidmNumberOfTrees );
	mQualityMetric = aSettings->value( "Optimizer/QualityMetric" ).toString().toLower();
	mMaxDepth = aSettings->value( "Optimizer/MaxDepth" ).toInt( &isValidmMaxDepth );
	mMinSamplesAtLeaf = aSettings->value( "Optimizer/MinSamplesAtLeaf" ).toInt( &isValidmMinSamplesAtLeaf );
	mKDEAttributesPerSplit = aSettings->value( "Optimizer/KDEAttributesPerSplit" ).toInt( &isValidmKDEAttributesPerSplit );
	mTreeSelection = aSettings->value( "Optimizer/TreeSelection" ).toString().toLower();
	mNumberSelectedTrees = aSettings->value( "Optimizer/NumberSelectedTrees" ).toInt( &isValidmNumberSelectedTrees );
	mBaggingMethod = aSettings->value( "Optimizer/BaggingMethod" ).toString().toLower();
	mBagFraction = aSettings->value( "Optimizer/BagFraction" ).toDouble( &isValidmBagFraction );
	mBoosting = aSettings->value( "Optimizer/Boosting" ).toString().toLower();

	if ( !isValidmNumberOfTrees || !isValidmMaxDepth || !isValidmMinSamplesAtLeaf || !isValidmKDEAttributesPerSplit || !isValidmNumberSelectedTrees || !isValidmBagFraction )
	{
		qDebug() << "Cannot read settings file for RandomForestOptimizer.";
	}

	auto labels = mDataPackage->labelOutcomes();
	if ( labels.contains( "NONE" ) )
	{
		qDebug() << "RandomForestOptimizer ERROR - Label names must not be NONE";
	}
}

//-----------------------------------------------------------------------------

RandomForestOptimizer::~RandomForestOptimizer()
{
	mBoostMultiplier.clear();
	mKeysToWeights.clear();
}

//-----------------------------------------------------------------------------

void RandomForestOptimizer::build()
{
	lpmleval::KernelDensityExtractor kde( mDataPackage->featureDatabase(), mDataPackage->labelDatabase(), mDataPackage->labelIndex() );
	QList< int > attributeIndex = kde.overlapRatios().values();	// Get attribute with lowest distribution overlap

	QVector < QPair< lpmldata::TabularData, lpmldata::TabularData > > subsamples = createRandomSubsample();

	auto rfModel = dynamic_cast< lpmleval::RandomForestModel* >( mModel );
	if ( rfModel == nullptr )
	{
		qDebug() << "RandomForestOptimizer ERROR - mModel is not a RandomForestModel";
		return;
	}
	
	rfModel->setSubsamples( subsamples );	// Needed in KDE tree selection and Evolving RandomForest
	QMap< double, DecisionTreeModel* > decisionTreeModelScores;	// Only needed in OOB tree selection. double = performance score

	if ( mBoosting != "None" && mBoostMultiplier.isEmpty() )	initializeBoostMultiplier();	// Initialize boosting multiplier to 1

	/*QTime timer;
	timer.start();*/
	unsigned int treeCount = 0;
	const unsigned int totalTreeCount = subsamples.size();

	int bagIndex = 0;
	for ( auto pair : subsamples )
	{
		// Create tree models
		lpmldata::DataPackage* subData = new lpmldata::DataPackage( pair.first, pair.second, mDataPackage->labelName() );
		
		lpmleval::DecisionTreeOptimizer* tree = new lpmleval::DecisionTreeOptimizer( mSettings, subData );
		tree->setInstanceWeights( mKeysToWeights[ bagIndex ] );

		if ( mBoosting != "None" )	tree->updateWeights( mKeysToWeights[ bagIndex ], mBoostMultiplier );

		tree->build();
		lpmleval::DecisionTreeModel* treeModel = dynamic_cast< DecisionTreeModel* > ( tree->model() );

		delete tree;
		tree = nullptr;
		delete subData;
		subData = nullptr;

		treeCount++;

		//qDebug() << "Created Tree " << treeCount << "/" << totalTreeCount << " Time: " << timer.elapsed();

		// Text-based model visualization
		//TreeIO io;
		//io.visualizeModel(treeModel, mTrainingFeatureSet.headerNames().toVector() );

		// Update weights for next tree if there are more tree to build
		if ( mBoosting == "adaboost" && bagIndex < subsamples.size() - 1 )	calculateBoostMultiplier( treeModel, bagIndex );

		if ( mTreeSelection == "oob" )	decisionTreeModelScores = selectBestTreesByOOB( treeModel, decisionTreeModelScores, pair.first );
		else if ( treeModel != nullptr )		rfModel->addDecisionTreeModel( treeModel );

		bagIndex++;
	}

	if ( mTreeSelection == "oob" )
	{
		for ( auto model : decisionTreeModelScores.values() )
		{
			rfModel->addDecisionTreeModel( model );
		}
	}

	//if ( mModel != nullptr )
	//{
	//	delete mModel;
	//	mModel = nullptr;
	//}

	//mModel = rfModel;

	mAnalytics->setDataPackage( mDataPackage );
	mAnalytics->evaluate( mModel );
}

//-----------------------------------------------------------------------------

void RandomForestOptimizer::calculateBoostMultiplier( lpmleval::DecisionTreeModel* aTreeModel, const int aBagIndex )
{
	// Calculate error
	double weightsInvalidSamples = 0;
	double weightsAllSamples = 0;
	QMap< QVariant, bool > samplePredictionIsValid;

	for ( auto key : mDataPackage->sampleKeys() )
	{
		QVariant trueLabel = mDataPackage->labelDatabase().valueAt( key, mDataPackage->labelIndex() );
		QVector< double > featureVector;
		for ( auto feature : mDataPackage->featureDatabase().value( key ) )
		{
			featureVector.push_back( feature.toDouble() );
		}
		QVariant predictedLabel = aTreeModel->evaluate( featureVector );

		if ( predictedLabel != trueLabel )
		{
			weightsInvalidSamples += mKeysToWeights[ aBagIndex + 1 ][ key ];
			samplePredictionIsValid[ key ] = false;
		}
		else
		{
			samplePredictionIsValid[ key ] = true;
		}

		weightsAllSamples += mKeysToWeights[ aBagIndex + 1 ][ key ];
	}

	double error = weightsInvalidSamples / weightsAllSamples;

	// Calculate alpha
	double alpha = 0.5 * log( ( 1 - error ) / error );

	// Modify weights
	for ( auto key : samplePredictionIsValid.keys() )
	{
		if ( samplePredictionIsValid[ key ] == true )
		{
			mBoostMultiplier[ key ] = mBoostMultiplier[ key ] * exp( -alpha );
		}
		else
		{
			mBoostMultiplier[ key ] = mBoostMultiplier[ key ] * exp( alpha );
		}
	}

	// Normalize
	double totalMultiplier = 0;
	for ( auto key : mBoostMultiplier.keys() )
	{
		totalMultiplier += mBoostMultiplier[ key ];
	}

	for ( auto key : mBoostMultiplier.keys() )
	{
		mBoostMultiplier[ key ] = mBoostMultiplier[ key ] / totalMultiplier;
	}
}

//-----------------------------------------------------------------------------

QPair< lpmldata::TabularData, lpmldata::TabularData > RandomForestOptimizer::createOOBSample( const lpmldata::TabularData& aInBagSamples )
{
	lpmldata::TabularData OOBFeatures = mDataPackage->featureDatabase();
	lpmldata::TabularData OOBLabels = mDataPackage->labelDatabase();

	for ( auto key : mDataPackage->sampleKeys() )
	{
		if ( aInBagSamples.keys().contains( key ) )
		{
			OOBFeatures.remove( key );
			OOBLabels.remove( key );
		}
	}

	return{ OOBFeatures, OOBLabels };
}

//-----------------------------------------------------------------------------

QVector < QPair< lpmldata::TabularData, lpmldata::TabularData > > RandomForestOptimizer::createRandomSubsample()
{
	lpmleval::TabularDataFilter filter;
	QVector < QPair< lpmldata::TabularData, lpmldata::TabularData > > subsamples;
	QVector< QMap < QVariant, double > > keysToWeights;
	//if ( mBaggingMethod == "stratified" )
	//{
	//	keysToWeights = filter.stratifiedBagging( mTrainingLabelSet, mNumberOfTrees, mBagFraction );
	//}
	if ( mBaggingMethod == "normal" )
	{
		keysToWeights = filter.bagging( mDataPackage, mNumberOfTrees, mBagFraction );
	}
	//else if ( mBaggingMethod == "walker" )
	//{
	//	keysToWeights = filter.walkerBagging( mTrainingLabelSet, mNumberOfTrees, mBagFraction );
	//}
	else if ( mBaggingMethod == "equalized" )
	{
		keysToWeights = filter.equalizedBagging( mDataPackage, mNumberOfTrees, mBagFraction );
	}
	else
	{
		qDebug() << "Error: Bagging method " << mBaggingMethod << " unkown";
	}

	mKeysToWeights = keysToWeights;

	for ( auto bag : keysToWeights )
	{
		auto keys = bag.keys();

		QStringList keysStr;
		for ( int i = 0; i < keys.size(); ++i )
		{
			keysStr.push_back( keys.at( i ).toString() );
		}
		QPair< lpmldata::TabularData, lpmldata::TabularData > setPair( filter.subTableByKeys( mDataPackage->featureDatabase(), keysStr ), filter.subTableByKeys( mDataPackage->labelDatabase(), keysStr ) );
		subsamples.append( setPair );
	}

	return subsamples;
}

//-----------------------------------------------------------------------------

void RandomForestOptimizer::initializeBoostMultiplier()
{
	auto keys = mDataPackage->sampleKeys();
	for ( auto key : keys )
	{
		mBoostMultiplier[ key ] = 1.0 / double( keys.size() );
	}
}

//-----------------------------------------------------------------------------

QVector< double > RandomForestOptimizer::normalize( QVector< double > aValues )
{
	double numberValues = aValues.size();
	for ( int index = 0; index < aValues.size(); index++ )
	{
		aValues[ index ] = aValues[ index ] / numberValues;
	}

	return aValues;
}

//-----------------------------------------------------------------------------

QMap< double, lpmleval::DecisionTreeModel* > RandomForestOptimizer::selectBestTreesByOOB(
	lpmleval::DecisionTreeModel* aTreeModel, QMap< double, lpmleval::DecisionTreeModel* > aTreeScores, const lpmldata::TabularData& aInBagSamples )
{
	lpmldata::TabularData OOBFeatures = createOOBSample( aInBagSamples ).first;

	double score;

	score = mAnalytics->evaluate( aTreeModel );

	if ( aTreeScores.size() < mNumberSelectedTrees )
	{
		aTreeScores[ score ] = aTreeModel;
	}
	else
	{
		for ( auto element : aTreeScores.keys() )
		{
			if ( score > element )
			{
				aTreeScores[ score ] = aTreeModel;
				aTreeScores.erase( aTreeScores.begin() );
				break;
			}
		}
	}

	return aTreeScores;
}

//-----------------------------------------------------------------------------

}
