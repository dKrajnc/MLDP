#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractModel.h>
#include <Evaluation/PipelineTree.h>
#include <DataRepresentation/DataPackage.h>
#include <Evaluation/AbstractTDPAction.h>
#include <DataRepresentation/DataPackage.h>
#include <QDir>

namespace dkeval
{

//-----------------------------------------------------------------------------


class Evaluation_API PipelineModel : public lpmleval::AbstractModel
{

public:

	PipelineModel( QSettings* aPluginModelSettings, QVector< std::shared_ptr< Node > > aPipeline, lpmldata::DataPackage& aDataPackage );


	void set( const QVector< double >& aParameters ) override;

	QVariant evaluate( const QVector< double >& aFeatureVector ) override;

	QVector< std::shared_ptr< dkeval::AbstractTBPAction > > dpactions() { return mDPActions; }

	int inputCount() override;

	AbstractModel* model() { return mPluginModel; }

	~PipelineModel();

	void save( QDataStream& aOut ) override {};

	void load( QDataStream& aIn ) override {};

	double fitness() { return mFitness; }

	void setFoldId( const int& aFoldId ) { mFoldId = aFoldId; }

private:

	void clearCache();

private:

	QVector< std::shared_ptr< dkeval::AbstractTBPAction > > mDPActions;
	QVector< std::shared_ptr< Node > > mPipeline;
	AbstractModel* mPluginModel;
	lpmldata::DataPackage mDataPackage;
	QMap< QString, QVariantList > mRanges;
	double mFitness;
	int mFoldId;
};

//-----------------------------------------------------------------------------

}

