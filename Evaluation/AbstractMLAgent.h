#pragma once

#include <Evaluation/Export.h>
#include <DataRepresentation/DataPackage.h>
#include <Evaluation/AbstractOptimizer.h>
#include <Evaluation/AbstractModel.h>
#include <Evaluation/AbstractAnalytics.h>
#include <QSettings>
#include <QString>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API AbstractMLAgent
{

public:

	AbstractMLAgent( QSettings* aSettings );

	virtual void train();

	virtual void validate( lpmldata::DataPackage* aDataPackage );

	virtual const QVariant evaluate( QVector< double > aFeatureVector );

	virtual ~AbstractMLAgent();

	const QString& uid() const { return mMLUID; }
	const QString& complexity() const { return mComplexity; }
	lpmldata::DataPackage* dataPackage() { return mDataPackage; }
	lpmleval::AbstractOptimizer* optimizer() { return mOptimizer; }
	lpmleval::AbstractModel* model() { return mModel; }
	lpmleval::AbstractAnalytics* analytics() { return mAnalytics; }
	const QString& modelType() const { return mModelType; }

	virtual void save() = 0;

	virtual void load() = 0;

protected:

	AbstractMLAgent();

	void generateMLAgents();

	lpmldata::DataPackage* buildMetaTable( lpmldata::DataPackage* aDataPackage );

protected:

	QSettings*                    mSettings;
	lpmldata::DataPackage*        mDataPackage;
	lpmleval::AbstractOptimizer*  mOptimizer;
	lpmleval::AbstractModel*      mModel;
	lpmleval::AbstractAnalytics*  mAnalytics;
	QString                       mMLUID;
	QString                       mComplexity;
	QString                       mDataPackageType;
	QString                       mAnalyticsType;
	QString                       mModelType;
	QString                       mOptimizerType;
	QList< AbstractMLAgent* >     mInputMLAgents;
};

//-----------------------------------------------------------------------------

}
