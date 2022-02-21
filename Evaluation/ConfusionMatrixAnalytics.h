#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractAnalytics.h>
#include <DataRepresentation/Array2D.h>
#include <DataRepresentation/TabularData.h>

namespace lpmleval
{

enum class ConfusionMatrixMeasure
{
	ROC = 0,
	FScore,
	AUC,
	ACC,
	SNS,
	SPC,
	PPV,
	NPV,
	MCC
};

//-----------------------------------------------------------------------------

class Evaluation_API ConfusionMatrixAnalytics: public AbstractAnalytics
{

public:

	ConfusionMatrixAnalytics( QSettings* aSettings, lpmldata::DataPackage* aDataPackage );

	virtual ~ConfusionMatrixAnalytics();

	double evaluate( lpmleval::AbstractModel* aModel ) override;
	double rocDistance();
	double fScore( double aBeta = 2.0 );
	double auc();
	double acc();
	double sns();
	double spc();
	double ppv();
	double npv();
	double mcc();
	QMap< QString, double > allValues() override;
	const QString& unit() override { return mUnit; }
	QMap< QString, double > confusionMatrixElements(); //added by Denis

private:

	ConfusionMatrixAnalytics();
	void calculateAtomicErrors();
	void resetConfusionMatrix();
	void resetContainers();

private:

	lpmldata::Array2D< unsigned int >*  mConfusionMatrix;
	ConfusionMatrixMeasure              mConfusionMatrixMeasure;
	QVector< double >                   mTPs;
	QVector< double >                   mTNs;
	QVector< double >                   mFPs;
	QVector< double >                   mFNs;
	double                              mDiagonalSum;
	double                              mSum;
	QString                             mUnit;
	bool                                mIsValid;
};

//-----------------------------------------------------------------------------

}
