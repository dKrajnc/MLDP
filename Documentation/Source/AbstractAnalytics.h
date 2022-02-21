#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractModel.h>
#include <DataRepresentation/DataPackage.h>
#include <QSettings>
#include <QString>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API AbstractAnalytics
{

public:

	AbstractAnalytics( QSettings* aSettings, lpmldata::DataPackage* aDataPackage ): mSettings( aSettings ), mDataPackage( aDataPackage ) {}

	virtual double evaluate( lpmleval::AbstractModel* aModel ) = 0;

	virtual ~AbstractAnalytics();

	virtual void setDataPackage( lpmldata::DataPackage* aDataPackage ) { /*mDataPackage = aDataPackage;*/ }

	virtual const QString& unit() = 0;

	virtual QMap< QString, double > allValues() = 0;

private:

	AbstractAnalytics();

protected:

	QSettings*              mSettings;
	lpmldata::DataPackage*  mDataPackage;
};

//-----------------------------------------------------------------------------

}
