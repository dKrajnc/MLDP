/*!
* \file
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#include <Evaluation/Export.h>
#include <DataRepresentation/TabularData.h>
#include <Evaluation/CovarianceMatrix.h>
#include <Evaluation/KernelDensityExtractor.h>

#include <QVector>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API FeatureSelector
{

public:

	FeatureSelector( lpmldata::TabularData aFeatureDatabase, lpmldata::TabularData aLabelDatabase, int aLabelIndex );

	~FeatureSelector();

	QVector< double > executeFrCovMxGlobal( double aCovarianceThreshold );

	QVector< double > executeFrCovMxLocal( double aCovarianceThreshold );

	QVector< double > executeFsRandom( double aChanceToSelect );

	QVector< double > executeFsKdeOverlap( int aSelectedFeatureCount, int aBootstrapSize = -1 );

private:

	FeatureSelector() {};

private:
	lpmldata::TabularData                     mFeatureDatabase;
	lpmldata::TabularData                     mLabelDatabase;
	int                                       mLabelIndex;
	int                                       mFeatureCount;
	std::random_device                        mRd;
	std::mt19937                              mEng;
	std::uniform_real_distribution< double >  mRealDistribution;
};

//-----------------------------------------------------------------------------

}