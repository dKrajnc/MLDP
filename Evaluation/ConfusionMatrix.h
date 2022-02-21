#pragma once

#include <Evaluation/Export.h>
#include <DataRepresentation/Array2D.h>
#include <QVector>

namespace lpmleval
{

//-----------------------------------------------------------------------------


class Evaluation_API ConfusionMatrix: public lpmldata::Array2D< unsigned int >
{

public:

	ConfusionMatrix( unsigned int aClassifierCount );

	virtual ~ConfusionMatrix();

	void evaluate();

	double UWF( double aRatio = 1.0 );

	double sensitivity() { return mAvgTPR; }

	double specificity() { return 1.0 - mAvgFPR; }

	double roc() { return mRoc; }

	double npv() { return mNpv; }

	double accuracy() { return mAccuracy; }

	double accuracyLocalAveraged() { return mAccuracyLocalAvg; }

	const QList< QList< double > >& errorValues() const { return mErrorValues; }

	double userBinary() { return mUser; }

private:

	void initValues();

	double rocDistance( double aTP, double aTN, double aFP, double aFN );

	double rocDistance2( const QList< double >& aTPRs );

private:

	double                    mAvgTP;				  
	double                    mAvgTN;				  
	double                    mAvgFP;				  
	double                    mAvgFN;				  
	double                    mAvgTPR;			  
	double                    mAvgFPR;			  
	double                    mRoc;	
	double                    mNpv;
	double                    mAccuracy;			  
	double                    mAccuracyLocalAvg;    
	QList< QList< double > >  mErrorValues;
	double                    mUser;

};

//-----------------------------------------------------------------------------

}
