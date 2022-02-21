#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractModel.h>
#include <Evaluation/AbstractAnalytics.h>
#include <DataRepresentation/DataPackage.h>
#include <QString>

namespace dkeval
{

//-----------------------------------------------------------------------------

class Evaluation_API PipelineAnalytics: public lpmleval::AbstractAnalytics
{

public:

	PipelineAnalytics( QSettings* aSettings, lpmldata::DataPackage* aDataPackage, lpmleval::AbstractModel* aModel ) : AbstractAnalytics( nullptr, aDataPackage ), mModel( aModel ), mSettings( aSettings ), mPreprocessedData(*aDataPackage){}
		

	double evaluate( lpmleval::AbstractModel* aModel ) override;

	void setDataPackage( lpmldata::DataPackage* aDataPackage ) override;

	~PipelineAnalytics();

	const QString& unit() { return "PipelineAnalytics"; };

	QMap< QString, double > allValues() override;

	std::shared_ptr< lpmldata::DataPackage > preProcessedDataPackage() { std::shared_ptr< lpmldata::DataPackage > preprocessedData = std::make_shared< lpmldata::DataPackage >( mPreprocessedData ); return preprocessedData; }

private:

	PipelineAnalytics();

private:
	lpmleval::AbstractModel* mModel;
	QSettings* mSettings;
	lpmldata::DataPackage mPreprocessedData;
};

//-----------------------------------------------------------------------------

}
