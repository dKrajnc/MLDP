/*!
* \file
* FeatureKernel class defitition. This file is part of Evaluation module.
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#include <Evaluation/Export.h>
#include <QVariant>
#include <QVector>


namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API FeatureKernel
{

public:

	FeatureKernel( const QVariantList& aFeatureEntries );
	FeatureKernel( const QVariantList& aFeatureEntries, double aRangeMin, double aRangeMax );

	~FeatureKernel();

	void execute();
	double fitness( double aFeature ) const;
	QVector< double > render( double aRangeMin, double aRangeMax ) const;

	double minReal() { return toReal( mFeatureKernelArray.at( 0 ) ); }
	double maxReal() { return toReal( mFeatureKernelArray.at( mFeatureKernelArray.size() - 1 ) ); }
	double maxDensity();

	double kernelDensityMaximum() const { return mKernelDensityMaximum; }
	void normalize( double aMaximum );

	inline int toIndex( double aReal ) const
	{
		return ( ( ( aReal - mMin ) / mRange ) * mBinSize ) + mBinMargin;
	}

	inline double toReal( int aIndex ) const
	{
		return ( double( double( aIndex - mBinMargin ) / double( mBinSize ) ) * mRange ) + mMin;
	}

	double entryMinimum() { return mMin; }
	double entryMaximum() { return mMax; }

private:

	void addEntries( const QVariantList& aFeatureEntries );
	void initSigmaAndMargin();
	void addAtomicFeatureGaussian( double aFeature );
	void initFeatureKernelArray();
	void clear() { mFeatureEntries.clear(); }
	double sampleSigma();
	
private:

	QVector< double >  mFeatureEntries;
	QVector< double >  mFeatureKernelArray;
	double             mBinWidth;
	int                mBinSize;
	int                mBinMargin;
	double             mSigma;
	double             mMargin;
	double             mMin;
	double             mMax;
	double             mRange;
	double             mGaussianDenominator;
	double             mKernelDensityMaximum;
	bool               mIsManualRange;

};

//-----------------------------------------------------------------------------

}
