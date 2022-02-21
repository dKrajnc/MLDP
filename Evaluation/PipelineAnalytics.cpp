#include <Evaluation/PipelineAnalytics.h>
#include <Evaluation/PipelineModel.h>
#include <Evaluation/ConfusionMatrixAnalytics.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

double PipelineAnalytics::evaluate( lpmleval::AbstractModel* aModel )
{
	lpmleval::ConfusionMatrixAnalytics cm( mSettings, mDataPackage ); //Added mSettings instead of nullptr

	return cm.evaluate( aModel );
}

//-----------------------------------------------------------------------------

QMap< QString, double > PipelineAnalytics::allValues()
{
	QMap< QString, double > result;

	lpmleval::ConfusionMatrixAnalytics cm( nullptr, mDataPackage );

	cm.evaluate( mModel );

	result = cm.allValues();

	return result;
}

//-----------------------------------------------------------------------------

void PipelineAnalytics::setDataPackage( lpmldata::DataPackage* aDataPackage )
{
	auto pipelineModel = dynamic_cast< dkeval::PipelineModel* > ( mModel );
	lpmldata::DataPackage currentDataPackage = *aDataPackage;

	if ( pipelineModel != nullptr )
	{
		auto pipeline = pipelineModel->dpactions();

		for ( auto dpaction : pipeline )
		{
			currentDataPackage = dpaction->run( currentDataPackage );
		}
	}
	else
	{
		qDebug() << "Warning - PipelineModel is nullptr!";
	}

	mPreprocessedData = currentDataPackage;
	//*mDataPackage = currentDataPackage;
}

//-----------------------------------------------------------------------------

dkeval::PipelineAnalytics::~PipelineAnalytics()
{
}

//-----------------------------------------------------------------------------

}