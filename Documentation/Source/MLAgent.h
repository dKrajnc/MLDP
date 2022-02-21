#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractMLAgent.h>
#include <Evaluation/ConfusionMatrixAnalytics.h>
#include <Evaluation/RandomForestOptimizer.h>
#include <Evaluation/RandomForestModel.h>
#include <DataRepresentation/DataPackage.h>
#include <FileIo/TabularDataFileIo.h>
#include <QFile>
#include <QDir>
#include <QDateTime>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API MLAgent: public AbstractMLAgent
{

public:

	MLAgent( QSettings* aSettings )
	:
		AbstractMLAgent( aSettings )
	{
		
		generateDataPackage();
		generateAnalytics();
		generateModel();
		generateOptimizer();

		/*qDebug() << "-----------------------------------------------------------------";
		qDebug() << "MLAgent with MLUID" << mMLUID << "generated.";
		qDebug() << "-----------------------------------------------------------------";*/
	}

	void generateDataPackage()
	{
		if ( mComplexity == "Atomic" )
		{
			auto FDBPath = mSettings->value( "FDBPath" ).toString();
			auto LDBPath = mSettings->value( "LDBPath" ).toString();

			lpmlfio::TabularDataFileIo loader;
			lpmldata::TabularData FDB;
			lpmldata::TabularData LDB;

			loader.load( FDBPath, FDB );
			loader.load( LDBPath, LDB );

			mDataPackage = new lpmldata::DataPackage( FDB, LDB );
		}
		if ( mComplexity == "Complex" )
		{
			/*lpmleval::DataPackage* inputDataPackage = new DataPackage( mSettings );
			mDataPackage = buildMetaTable( inputDataPackage );
			delete inputDataPackage;*/
		}
	}

	void generateAnalytics()
	{	
		if ( mAnalyticsType == "ConfusionMatrixAnalytics" )
		{
			mAnalytics = new lpmleval::ConfusionMatrixAnalytics( mSettings, mDataPackage );
		}
	}

	void generateModel()
	{
		if ( mModelType == "MaskModel" )
		{
			//mModel = new lpmleval::MaskModel( mSettings, mDataPackage );
		}
		else if ( mModelType == "RandomForestModel" )
		{
			mModel = new lpmleval::RandomForestModel( mSettings );
		}
	}

	void generateOptimizer()
	{
		if ( mOptimizerType == "RandomForestOptimizer" )
		{
			mOptimizer = new lpmleval::RandomForestOptimizer( mSettings, mDataPackage, mModel, mAnalytics );
		}
		else if ( mOptimizerType == "GeneticAlgorithmOptimizer" )
		{
			//mOptimizer = new lpmleval::GeneticAlgorithmOptimizer( mSettings, mDataPackage, mModel, mAnalytics );
		}
	}

	void save()
	{
		QString fileName = mSettings->fileName();
		QString MLAgentPath = fileName.mid( 0, fileName.lastIndexOf( "/" ) );
		QString MLAgentLogPath = MLAgentPath + "/Snapshot";
		if ( !QDir().exists( MLAgentLogPath ) )
		{
			QDir().mkpath( MLAgentLogPath );
		}
		QString snapshotFileName = QDateTime().currentDateTimeUtc().toString();
		snapshotFileName = snapshotFileName.replace( ":", "-" );
		QString snapshotFilePath = MLAgentLogPath + "/" + snapshotFileName + ".txt";

		QFile file( snapshotFilePath );

		if ( !file.open(QIODevice::WriteOnly ) )
		{
			qDebug() << "Could not open file to save: " << snapshotFilePath;
		}
		else
		{
			QDataStream out( &file );
			out.setVersion( QDataStream::Qt_5_5 );

			mModel->save( out );

			file.flush();
			file.close();

			mSettings->sync();

			// Copy snapshot to latest.txt
			QString laterstFilePath = MLAgentLogPath + "/latest.txt";
			if ( QFile::exists( laterstFilePath ) )
			{
				QFile::remove( laterstFilePath );
			}
			QFile::copy( snapshotFilePath, laterstFilePath );
		}
	}

	void load()
	{
		QString fileName = mSettings->fileName();
		QString MLAgentPath = fileName.mid( 0, fileName.lastIndexOf( "/" ) );
		QString MLAgentLogPath = MLAgentPath + "/Snapshot";
		QString snapshotFilePath = MLAgentLogPath + "/latest.txt";
		QFile file( snapshotFilePath );

		if ( !file.open( QIODevice::ReadOnly ) )
		{
			qDebug() << "MLAgent - LOAD: No previous snapshot found here: " << snapshotFilePath;
		}
		else
		{
			QDataStream in( &file );
			in.setVersion( QDataStream::Qt_5_5 );

			mModel->load( in );

			file.close();

			mAnalytics->evaluate( mModel );
		}
	}

	~MLAgent()
	{
		mSettings->sync();

		delete mOptimizer;
		delete mModel;
		delete mAnalytics;
		delete mDataPackage;
		delete mSettings;

		mOptimizer = nullptr;
		mModel = nullptr;
		mAnalytics = nullptr;
		mDataPackage = nullptr;
		mSettings = nullptr;

		//qDebug() << "-----------------------------------------------------------------";
		//qDebug() << "MLAgent with MLUID" << mMLUID << "terminated.";
		//qDebug() << "-----------------------------------------------------------------";
	}

private:

	MLAgent();
};

//-----------------------------------------------------------------------------

}
