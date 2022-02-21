#include <Evaluation/FeatureKernel.h>
#include <QDebug>

namespace lpmleval
{

//-----------------------------------------------------------------------------

FeatureKernel::FeatureKernel( const QVariantList& aFeatureEntries )
:
	mFeatureEntries(),
	mFeatureKernelArray(),
	mBinWidth( 0.0 ),
	mBinSize( 1000 ),
	mBinMargin( 0 ),
	mSigma( 0.0 ),
	mMargin( 0.0 ),
	mMin( DBL_MAX ),
	mMax( -DBL_MAX ),
	mRange( 0.0 ),
	mGaussianDenominator( 0.0 ),
	mKernelDensityMaximum( -DBL_MAX ),
	mIsManualRange( false )
{
	addEntries( aFeatureEntries );
	execute();
}

//-----------------------------------------------------------------------------

FeatureKernel::FeatureKernel( const QVariantList& aFeatureEntries, double aRangeMin, double aRangeMax )
:
	mFeatureEntries(),
	mFeatureKernelArray(),
	mBinWidth( 0.0 ),
	mBinSize( 1000 ),
	mBinMargin( 0 ),
	mSigma( 0.0 ),
	mMargin( 0.0 ),
	mMin( aRangeMin ),
	mMax( aRangeMax ),
	mRange( 0.0 ),
	mGaussianDenominator( 0.0 ),
	mKernelDensityMaximum( -DBL_MAX ),
	mIsManualRange( true )
{
	addEntries( aFeatureEntries );
	execute();
}

//-----------------------------------------------------------------------------

FeatureKernel::~FeatureKernel()
{
}

//-----------------------------------------------------------------------------

void FeatureKernel::addEntries( const QVariantList& aFeatureEntries )
{
	if ( mIsManualRange )
	{
		for ( int featureIndex = 0; featureIndex < aFeatureEntries.size(); ++featureIndex )
		{
			double feature = aFeatureEntries.at( featureIndex ).toDouble();
			mFeatureEntries.push_back( feature );
		}
	}
	else
	{
		for ( int featureIndex = 0; featureIndex < aFeatureEntries.size(); ++featureIndex )
		{
			double feature = aFeatureEntries.at( featureIndex ).toDouble();
			mFeatureEntries.push_back( feature );

			if ( feature < mMin )
			{
				mMin = feature;
			}
			else if ( feature > mMax )
			{
				mMax = feature;
			}
		}
	}
	
	mRange = mMax - mMin;

	std::sort( mFeatureEntries.begin(), mFeatureEntries.end() );
}

//-----------------------------------------------------------------------------

void FeatureKernel::execute()
{
	// Determine the atomic Gaussian kernel sigma.
	initSigmaAndMargin();

	// Set up variables based on sigma and range.
	if ( mRange == 0.0 )
	{
		mRange = 1.0;
		mBinWidth = 0.0;
		mBinMargin = 1;
	}
	else
	{
		mBinWidth = double( ( mMax - mMin ) / double( mBinSize ) );
		mBinMargin = std::max( int( mMargin / mBinWidth ), 1 );
	}

	// Set up the array size
	initFeatureKernelArray();

	// Create the mFeatureKernelArray.
	for ( int featureIndex = 0; featureIndex < mFeatureEntries.size(); ++featureIndex )
	{
		double feature = mFeatureEntries.at( featureIndex );
		addAtomicFeatureGaussian( mFeatureEntries.at( featureIndex ) );
	}
}

//-----------------------------------------------------------------------------

double FeatureKernel::fitness( double aFeature ) const
{
	int arrayIndex = toIndex( aFeature );

	if ( arrayIndex < 0 || arrayIndex > mFeatureKernelArray.size() -1 ) return 0.0;

	// TODO: Linear interpolation?
	return mFeatureKernelArray.at( arrayIndex );
}

//-----------------------------------------------------------------------------

void FeatureKernel::initSigmaAndMargin()
{	
	double sampleSD = sampleSigma() + DBL_EPSILON;

	int Q1Index = mFeatureEntries.size() * 0.25;
	int Q3Index = mFeatureEntries.size() * 0.75;

	double Q1 = mFeatureEntries.at( Q1Index );
	double Q3 = mFeatureEntries.at( Q3Index );

	double A = std::min( sampleSD, ( Q3 - Q1 ) / 1.34 ) + DBL_EPSILON;

	mSigma = 0.9 * A * std::pow( mFeatureEntries.size(), -0.2 );

	mMargin = 3.0 * mSigma;
	mGaussianDenominator = 2.0 * mSigma * mSigma;
}

//-----------------------------------------------------------------------------

void FeatureKernel::addAtomicFeatureGaussian( double aFeature )
{
	// Determine start and end position indices in mFeatureKernelArray.
	int indexOfFeature = toIndex( aFeature );

	int startIndex = std::max( indexOfFeature - mBinMargin, 0 );
	int endIndex   = std::min( indexOfFeature + mBinMargin, mFeatureKernelArray.size() - 1 );

	double mRecGaussianDenominator = 1.0 / mGaussianDenominator;

	// From start till end position, add the atomic gaussian evaluated values.
	for ( int arrayIndex = startIndex; arrayIndex <= endIndex; ++arrayIndex )
	{
		double realValue = toReal( arrayIndex );  // Determine the real value of the index.
		double gaussianValue = exp( -( ( ( aFeature - realValue )*( aFeature - realValue ) ) * mRecGaussianDenominator ) );  // determine its gaussian value.
		mFeatureKernelArray[ arrayIndex ] += gaussianValue;
	}
}

//-----------------------------------------------------------------------------

double FeatureKernel::maxDensity()
{
	mKernelDensityMaximum = -DBL_MAX;

	for ( int arrayIndex = 0; arrayIndex < mFeatureKernelArray.size(); ++arrayIndex )
	{
		if ( mFeatureKernelArray.at( arrayIndex ) > mKernelDensityMaximum ) mKernelDensityMaximum = mFeatureKernelArray.at( arrayIndex );
	}

	return mKernelDensityMaximum;
}

//-----------------------------------------------------------------------------

void FeatureKernel::normalize( double aMaximum )
{
		double recMax = 1.0 / aMaximum;

	for ( int arrayIndex = 0; arrayIndex < mFeatureKernelArray.size(); ++arrayIndex )
	{
		mFeatureKernelArray[ arrayIndex ] *= recMax;
	}
}

//-----------------------------------------------------------------------------

void FeatureKernel::initFeatureKernelArray()
{
	mFeatureKernelArray.clear();
	mFeatureKernelArray.resize( ( 2 * mBinMargin ) + mBinSize );
	mFeatureKernelArray.fill( 0.0 );
}

//-----------------------------------------------------------------------------

double FeatureKernel::sampleSigma()
{
	double mean = 0.0;
	double variance = 0.0;

	for ( int featureIndex = 0; featureIndex < mFeatureEntries.size(); ++featureIndex )
	{
		mean += mFeatureEntries.at( featureIndex );
	}

	mean /= mFeatureEntries.size();

	for ( int featureIndex = 0; featureIndex < mFeatureEntries.size(); ++featureIndex )
	{
		variance += std::pow( mFeatureEntries.at( featureIndex ) - mean, 2.0 );
	}

	variance /= mFeatureEntries.size();

	return std::sqrt( variance );
}

//-----------------------------------------------------------------------------

QVector< double > FeatureKernel::render( double aRangeMin, double aRangeMax ) const
{
	QVector< double > renderedKDE;
	double step = ( aRangeMax - aRangeMin ) / mBinSize;
	renderedKDE.resize( mBinSize );

	double feature = aRangeMin;

	for ( int i = 0; i < mBinSize; ++i )
	{
		renderedKDE[ i ] = fitness( feature );
		feature += step;
	}

	return renderedKDE;
}

//-----------------------------------------------------------------------------

}