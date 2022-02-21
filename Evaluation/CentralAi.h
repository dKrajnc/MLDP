/*!
* \file
* CentralAI class defitition. This file is part of Evaluation module.
* The CentralAI is a class for automated data preparation, based on evolutionary algorithm. It is responsible for pipeline generation, evaluation and hyperparameter optimization.
* \remarks
*
* \authors
* dkrajnc
*/

#pragma once

#include <QDebug>
#include <QDirIterator>
#include <qsettings.h>
#include <Evaluation/Export.h>
#include <Evaluation/PipelineTree.h>
#include <Evaluation/FeatureSelection.h>
#include <Evaluation/Oversampling.h>
#include <Evaluation/PCA.h>
#include <Evaluation/IsolationForest.h>
#include <Evaluation/Undersampling.h>
#include <Evaluation/ConfusionMatrixAnalytics.h>
#include <Evaluation/RandomForestModel.h>
#include <Evaluation/RandomForestOptimizer.h>
#include <Evaluation/PatientFoldGenerator.h>
#include <Evaluation/CMAnalytics.h>
#include <FileIo/TabularDataFileIo.h>
#include <random>
#include <fstream>


namespace dkeval
{

//-----------------------------------------------------------------------------

typedef QMap< double, Creature > Population;

struct PreprocessedPackage
{
	QVector< std::shared_ptr< dkeval::AbstractTBPAction > > tbpActions;
	std::shared_ptr < lpmldata::DataPackage > preprocessedDataPackage;
};

//-----------------------------------------------------------------------------

class Evaluation_API CentralAi 
{

public:

	/*!
	* \brief Constructor to build data preparation pipelines tree and to load settings parameters, plugin settings parameters, training dataset and validation dataset. 
	* \param [in] aSettings The settigns file of Central AI hyperparameters
	* \param [in] aSettings The settigns file of ML plugin hyperparameters
	* \param [in] aTrainingData The datapackage of training feature and label datasets
	* \param [in] aValidationData The datapackage of validation feature and label datasets
	*/
	CentralAi( QSettings* aSettings, QSettings* aPluginSettings, const lpmldata::DataPackage& aTrainingData, const lpmldata::DataPackage& aValidationData )
	:
		mRng( nullptr ),
		mSettings( aSettings ),
		mPluginSettings( aPluginSettings ),
		mIsInitValid( true ),
		mOffspringCount(),
		mMutationRate(),
		mIterationCount( 0 ),
		mPopulation(),
		mTrainingData( aTrainingData ),
		mValidationData( aValidationData ),
		mTBPAction(),
		mROC(),
		mOptimizedParameters(),
		mPreprocessedDataPackage( aTrainingData ),
		mConfusionMatrixValues(),
		mFoldId(),
		mPreprocessedDatasets()
	{
		if ( mSettings == nullptr )
		{
			qDebug() << "CentralAi - Error: Settings is a nullptr";
			mIsInitValid = false;
		}
		else
		{
			bool isOffspringCount;
			mOffspringCount = std::abs( mSettings->value( "CentralAi/offspringCount" ).toInt( &isOffspringCount ) );
			if ( !isOffspringCount )
			{
				qDebug() << "CentralAi - Error: Invalid parameter offspringCount";
				mIsInitValid = false;
			}

			bool isMutationRate;
			mMutationRate = std::abs( mSettings->value( "CentralAi/mutationRate" ).toDouble( &isMutationRate ) );
			if ( !isMutationRate )
			{
				qDebug() << "CentralAi - Error: Invalid parameter mutationRate";
				mIsInitValid = false;
			}

			bool isIterationCount;
			mIterationCount = std::abs( mSettings->value( "CentralAi/iterationCount" ).toInt( &isIterationCount ) );
			if ( !isIterationCount )
			{
				qDebug() << "CentralAi - Error: Invalid parameter iterationCount";
				mIsInitValid = false;
			}
		}		

		mTree = new dkeval::PipelineTree( aSettings );
		mTree->buildTree();

		std::random_device rd;
		mRng = new std::mt19937( rd() );
	};
	
	/*!
	* \brief Destructor
	*/
	~CentralAi() { delete mRng; delete mTree; }	

	
public:

	/*!
	* \brief Runs the automated data preparation engine
	*/
	void execute();

	/*!
	* \brief Get confusion matrix values
	* \return QMap< QString, double > of confussion matrix names and values
	*/	
	QMap< QString, double > confusionMatrixValues() { return mConfusionMatrixValues; }

	/*!
	* \brief Fold saver
	* \param [in] aFoldPath The path to save location
	*/
	void saveFoldAt( QString aFoldPath );

	/*!
	* \brief Data preparation pipeline information file saver
	* \param [in] aFoldPath The path to save location
	* \param [in] aFileName The name of saved file
	*/
	void savePipelineInfo( QString aFoldPath, QString aFileName );

	/*!
	* \brief Performance information file saver
	* \param [in] aFoldPath The path to save location
	* \param [in] aFileName The name of saved file
	*/
	void savePerformanceInfo( QString aFoldPath, QString aFileName );

	/*!
	* \brief Set the ID of analyzed fold
	* \param [in] aFoldId The unique fold ID
	*/
	void setFoldId( const int& aFoldId ) { mFoldId = aFoldId; };

private:
	
	CentralAi();
	void iteratePopulation();
	Creature offspring( Creature aParent_1, Creature aParent_2 );
	QString chooseParent();
	QString chooseChild( QList< QString >& aChildren );	
	std::shared_ptr< Node > mutate( std::shared_ptr< Node > aOffspring, QVector< std::shared_ptr< Node > >& aSiblings );
	void removeIfContains( QList< QString >& aList, QString aElement );
	std::shared_ptr< Node > nodeIfAlgContains( QVector< std::shared_ptr< Node > >& aSiblings, std::shared_ptr< Node > aNode );
	void initializePopulation( const int& aNumberOfCreatures );
	QPair < Creature, Creature > parents( const Population& aPopulation );
	double calculateFitness( Creature aCreature, bool aIsTraining );
	lpmldata::DataPackage preProcessData( const lpmldata::DataPackage& aData );
	void evaluatePopulation();
	int randomIndex( int aListSize );
	double randomPercentage();

private:
	
	std::mt19937* mRng;
	dkeval::PipelineTree* mTree;
	QSettings* mSettings;
	QSettings* mPluginSettings;
	bool mIsInitValid;
	int mOffspringCount;
	double mMutationRate;
	int mIterationCount;
	Population mPopulation;
	lpmldata::DataPackage mTrainingData;
	lpmldata::DataPackage mValidationData;
	QVector< std::shared_ptr< dkeval::AbstractTBPAction > > mTBPAction;
	double mROC;
	QVector< double > mOptimizedParameters;
	lpmldata::DataPackage mPreprocessedDataPackage;
	QMap< QString, double > mConfusionMatrixValues;
	int mFoldId;
	QVector< dkeval::PreprocessedPackage > mPreprocessedDatasets;
};

}
