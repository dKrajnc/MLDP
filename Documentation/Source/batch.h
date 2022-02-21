/*!
* \file
* Batch file defitition. This file is part of Evaluation module.
* The batch file contains convinience functions as well as functions for structure organization puropse which do not belong to defined classes 
*
* \remarks
*
* \authors
* dKrajnc
*/

#pragma once

#include "Evaluation/DataOptimizer.h"
#include "Evaluation/CentralAi.h"

typedef QVector< QPair< std::shared_ptr< lpmldata::DataPackage >, std::shared_ptr< lpmldata::DataPackage > > > MCFolds;

//-----------------------------------------------------------------------------

/*!
* \brief Saves the fold at provided location
* \param [in] aFoldPath The path to location
* \param [in] aTrainingDataPackage The training datapackage
* \param [in] aValidationDataPackage The validation datapackage
*/
void saveFoldAt( QString aFoldPath, lpmldata::DataPackage aTrainingDataPackage, lpmldata::DataPackage aValidationDataPackage )
{
	lpmlfio::TabularDataFileIo fileIo;

	fileIo.save( aFoldPath + "/TDS.csv", aTrainingDataPackage.featureDatabase() );
	fileIo.save( aFoldPath + "/TLD.csv", aTrainingDataPackage.labelDatabase() );
	fileIo.save( aFoldPath + "/VDS.csv", aValidationDataPackage.featureDatabase() );
	fileIo.save( aFoldPath + "/VLD.csv", aValidationDataPackage.labelDatabase() );
}

//-----------------------------------------------------------------------------

/*!
* \brief Inspects the datasets to eliminate misalignments in sample keys, to fill in missing values, eliminate empty features etc.
* \param [in] aFDB The feature database
* \param [in] aLDB The label database
* \return lpmldata::DataPackage Optimized datapackage
*/
lpmldata::DataPackage optimizeData( const lpmldata::TabularData& aFDB, const lpmldata::TabularData& aLDB )
{
	dkeval::DataOptimizer optimizer( aFDB );
	optimizer.build();

	auto redundandFeatures = optimizer.redundandFeatures();
	auto optimizedFDB      = optimizer.optimizedFeatureDatabase();

	//Create DataPackage
	auto LDB = aLDB;
	lpmldata::DataPackage dataPackage( optimizedFDB, LDB );
	dataPackage.updateLDB();

	return dataPackage;
}

//-----------------------------------------------------------------------------

/*!
* \brief Generate folds
* \param [in] aSettingsPath Path to setings file
* \param [in] aDataPackage The datapackage for which folds are generated
* \return MCFolds Generated Monte Carlo cross-validation folds
*/
MCFolds getFolds( QString& aSettingsPath, const lpmldata::DataPackage& aDataPackage )
{
	QSettings settings( aSettingsPath, QSettings::IniFormat );

	//Check parameter validity
	bool isSplitPercentage;
	auto splitPercentage = std::abs( settings.value( "CentralAi/splitPercentage" ).toDouble( &isSplitPercentage ) );
	if ( !isSplitPercentage )
	{
		qDebug() << "CentralAi - Error: Invalid parameter splitPercentage";
		isSplitPercentage = false;
	}

	bool isFoldCount;
	auto foldCount = std::abs( settings.value( "CentralAi/foldCount" ).toInt( &isFoldCount ) );
	if ( !isFoldCount )
	{
		qDebug() << "CentralAi - Error: Invalid parameter foldCount";
		isFoldCount = false;
	}


	//Generate folds
	auto validationSize = aDataPackage.sampleCountOfPercentage( splitPercentage );

	dkeval::PatientFoldGenerator foldGenerator( aDataPackage, validationSize );
	foldGenerator.generate( foldCount );
	qDebug() << "Number of generated folds: " << foldCount;


	//Create Monte Carlo cross-validation folds list
	MCFolds folds;

	for ( int i = 0; i < foldCount; ++i )
	{
		auto fold = foldGenerator.fold( i );

		folds.push_back( fold );
	}

	return folds;
}

//-----------------------------------------------------------------------------

/*!
* \brief Saves the performance from confussion matrix
* \param [in] aTotalConfusionMatrixValues The confussion matrix values
* \param [in] aDataPath The path to the save location
* \param [in] aFileName The name of the saved file
*/
void overallPerformance( QMap< QString, double >& aTotalConfusionMatrixValues, const QString& aDataPath, const QString& aFileName )
{
	dkeval::CMAnalytics cmAnalytics( aTotalConfusionMatrixValues );
	cmAnalytics.savePerformanceAt( aDataPath, aFileName );

	qDebug() << "Central AI analysis finished!";

	qDebug() << "Model performance:";
	qDebug() << "ACC" << cmAnalytics.acc();
	qDebug() << "SNS" << cmAnalytics.sns();
	qDebug() << "SPC" << cmAnalytics.spc();
	qDebug() << "PPV" << cmAnalytics.ppv();
	qDebug() << "NPV" << cmAnalytics.npv();
}

//-----------------------------------------------------------------------------

/*!
* \brief Performs the automated data preparation over single center data
* \param [in] aGlobalSettingsPath The path to location of Settings.ini and pluginSettings.ini files
* \param [in] aDataPath The path to the location of the feature and label databases
*/
void singleCenterAnalysis( const QString& aGlobalSettingsPath, const QString& aDataPath )
{
	QDir dir( aDataPath );
	QString settingsPath       = aGlobalSettingsPath + "Settings.ini";
	QString pluginSettingsPath = aGlobalSettingsPath + "pluginSettings.ini";	
	int foldCounter            = 0; 	

	//Load feature and label database from .csv file
	lpmldata::TabularData FDB;
	lpmldata::TabularData LDB;
	lpmlfio::TabularDataFileIo loader;
	loader.load( aDataPath + "FDB.csv", FDB );
	loader.load( aDataPath + "LDB.csv", LDB );

	//Check FDB+LDB and generate folds
	auto optimizedData = optimizeData( FDB, LDB );
	auto folds         = getFolds( settingsPath, optimizedData );

	//Store TP, TN, FP, FN across all folds
	QMap< QString, double > totalConfusionMatrixValues;	
	
	
	//Run paprallel analysis - singlecenter data
#pragma omp parallel for schedule( static )
	for ( int i = 0; i < folds.size(); ++i )
	{
		QSettings settings( settingsPath, QSettings::IniFormat );
		QSettings pluginSettings( pluginSettingsPath, QSettings::IniFormat );

		auto fold               = folds.at( i );
		auto trainingDataPair   = fold.first;
		auto validationDataPair = fold.second;
		
		dkeval::CentralAi ai( &settings, &pluginSettings, *trainingDataPair, *validationDataPair );
		ai.setFoldId( i + 1 );
		ai.execute();		
		qInfo() << "Fold:" << i + 1 << "analysis in progress...";

#pragma omp critical
		{
			auto confusionValues = ai.confusionMatrixValues();
			for ( auto key : confusionValues.keys() )
			{
				totalConfusionMatrixValues[ key ] += confusionValues.value( key );
			}

			foldCounter++;
		}


		//Store folds		
		auto foldPath = aDataPath + "/Folds/Fold-" + QString::number( i + 1 );
		dir.mkpath( foldPath );		
		ai.saveFoldAt( foldPath );


		//Store predctionSummary.csv, performance.csv and pipeline_info.txt
		foldPath = aDataPath + "/Summary/fold_information/Fold-" + QString::number( i + 1 );
		dir.mkpath( foldPath );

		ai.savePipelineInfo( foldPath, "/pipeline_info.txt" );
		ai.savePerformanceInfo( foldPath, "/performance_info.csv" );		
		qInfo() << "Fold:" << i + 1 << "analysis finished and saved!";


		//Remove temp files
		QFile pipelineSettingsFile( aGlobalSettingsPath + QString::number( i + 1 ) + "pipelineSettings.ini" );
		pipelineSettingsFile.remove();


		//Report progress
		qInfo() << foldCounter << "/" << folds.size() << "folds finished!";
	}


	//Calculate and store overall performance
	overallPerformance( totalConfusionMatrixValues, aDataPath, "/centralAI_overall_performance_info.csv" );

}

//-----------------------------------------------------------------------------

/*!
* \brief Performs the automated data preparation over multiple center data
* \param [in] aGlobalSettingsPath The path to location of Settings.ini and pluginSettings.ini files
* \param [in] aDataPath The path to the location of the feature and label databases for training and validation
*/
void multipleCenterAnalysis( const QString& aGlobalSettingsPath, const QString& aDataPath )
{
	QDir dir( aDataPath );
	QString settingsPath       = aGlobalSettingsPath + "Settings.ini";
	QString pluginSettingsPath = aGlobalSettingsPath + "pluginSettings.ini";

	QSettings settings( settingsPath, QSettings::IniFormat );
	QSettings pluginSettings( pluginSettingsPath, QSettings::IniFormat );
	

	lpmldata::TabularData trainingFDB;
	lpmldata::TabularData trainingLDB;
	lpmldata::TabularData validationFDB;
	lpmldata::TabularData validationLDB;
	lpmlfio::TabularDataFileIo loader;

	loader.load( aDataPath + "VDS.csv", validationFDB );
	loader.load( aDataPath + "VLD.csv", validationLDB );
	loader.load( aDataPath + "TDS.csv", trainingFDB );
	loader.load( aDataPath + "TLD.csv", trainingLDB );		
	qInfo() << "Data is loaded!";	


	//Create training and validation DataPackage
	lpmldata::DataPackage trainData( trainingFDB, trainingLDB );
	lpmldata::DataPackage validateData( validationFDB, validationLDB );


	//Store TP, TN, FP, FN
	QMap< QString, double > totalConfusionMatrixValues;
	
	
	//Run analysis - singlecenter data
	dkeval::CentralAi ai( &settings, &pluginSettings, trainData, validateData );
	ai.setFoldId( 1 );
	ai.execute();
	qInfo() << "Central AI data analysis in progress...";


	auto confusionValues = ai.confusionMatrixValues();
	for ( auto key : confusionValues.keys() )
	{
		totalConfusionMatrixValues[ key ] += confusionValues.value( key );
	}	


	//Store folds
	auto foldPath = aDataPath + "/Folds/Fold-" + QString::number( 1 );
	dir.mkpath( foldPath );	
	ai.saveFoldAt( foldPath );


	//Store performance.csv and pipeline_info.txt
	foldPath = aDataPath + "/Summary/fold_information/Fold-" + QString::number( 1 );
	dir.mkpath( foldPath );
	
	
	ai.savePipelineInfo( foldPath, "/pipeline_info.txt" );
	ai.savePerformanceInfo( foldPath, "/performance_info.csv" );


	//Remove temp files
	QFile pipelineSettingsFile( aGlobalSettingsPath + QString::number( 1 ) + "pipelineSettings.ini" );
	pipelineSettingsFile.remove();


	//Calculate and store overall performance
	overallPerformance( totalConfusionMatrixValues, aDataPath, "/overall_performance_info.csv" );	
}

//-----------------------------------------------------------------------------

/*!
* \brief Runs the automated data preparation executions
* \param [in] *aArgv[] Array of the terminal input arguments
*/
void runMLDP( char *aArgv[] )
{
	QString globalSettingsPath = QString::fromStdString( std::string( aArgv[ 1 ] ) );
	QString dataPath           = QString::fromStdString( std::string( aArgv[ 2 ] ) );
	QString studyType          = QString::fromStdString( std::string( aArgv[ 3 ] ) );
	
	//Swtich for SINGLE/MULTI center analysis
	if ( studyType == "SINGLE" )
	{
		singleCenterAnalysis( globalSettingsPath, dataPath );
	}
	else if ( studyType == "MULTI" )
	{
		multipleCenterAnalysis( globalSettingsPath, dataPath );
	}
	else
	{
		qInfo() << "Error - invalid study type!";
		std::exit( EXIT_SUCCESS );
	}	
}

//-----------------------------------------------------------------------------