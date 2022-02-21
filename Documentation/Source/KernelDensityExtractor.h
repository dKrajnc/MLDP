/*!
* \file
* KernelDensityExtractor class defitition. This file is part of Evaluation module.
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/TabularDataFilter.h>
#include <Evaluation/FeatureKernel.h>
#include <QVariant>
#include <QSettings>
#include <QVector>
#include <QString>


namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API KernelDensityExtractor
{

public:

	KernelDensityExtractor( lpmldata::TabularData& aFDB, const lpmldata::TabularData& aLDB, int aLabelIndex );
	~KernelDensityExtractor();

	const QStringList& labelGroups() const { return mLabelGroups; }
	const QMap< QString, QString >& columnNames() const { return mColumnNames; }
	const QString columnName( int aFeatureIndex ) { return mColumnNames.value( QString::number( aFeatureIndex ) ); }

	double minimum( int aFeatureIndex ) { return mMinimums.value( QString::number( aFeatureIndex ) ); }
	double maximum( int aFeatureIndex ) { return mMaximums.value( QString::number( aFeatureIndex ) ); }

	int featureCount() const { return mFeatureCount; }
	FeatureKernel* kernel( QString aLabelOutcome, int aFeatureIndex );
	FeatureKernel* kernel( int  aLabelOutcomeIndex, int aFeatureIndex );

	const QMap< double, int >& overlapRatios() const { return mOverlapRatio; }

private:

	void calculateOverlapRatios();
	double areaUnderCurve( const FeatureKernel* aFeatureKernel, int aFeatureIndex );
	double intersect( FeatureKernel* aLeft, FeatureKernel* aRight, int aFeatureIndex );

private:

	QMap< QString, FeatureKernel* >  mKernelMap;
	lpmleval::TabularDataFilter      mFilter;
	QMap< QString, double >          mMinimums;
	QMap< QString, double >          mMaximums;
	QMap< QString, QString >         mColumnNames;
	int                              mFeatureCount;
	QStringList                      mLabelGroups;
	QMap< double, int >              mOverlapRatio;

};

//-----------------------------------------------------------------------------

}
