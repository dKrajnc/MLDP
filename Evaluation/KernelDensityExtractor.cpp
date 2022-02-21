#include <Evaluation/KernelDensityExtractor.h>
#include <QDebug>
#include <omp.h>

namespace lpmleval
{

//-----------------------------------------------------------------------------

KernelDensityExtractor::KernelDensityExtractor( lpmldata::TabularData& aFDB, const lpmldata::TabularData& aLDB, int aLabelIndex )
:
	mKernelMap(),
	mFilter(),
	mMinimums(),
	mMaximums(),
	mColumnNames(),
	mFeatureCount( aFDB.columnCount() ),
	mLabelGroups(),
	mOverlapRatio()
{
	QStringList commonKeys  = mFilter.commonKeys( aFDB, aLDB, aLabelIndex );

	if ( commonKeys.size() == 0 )
	{
		qDebug() << "KernelDensityExtractor ERROR - there are no common keys in FDB and LDB with labelIndex " << aLabelIndex;
		system( "pause" );
	}

	mLabelGroups = mFilter.labelGroups( aLDB, aLabelIndex );

	for ( int featureIndex = 0; featureIndex < mFeatureCount; ++featureIndex )  // Go through the columns of the subtable.
	{
		double kernelMax = 0.0;

		// Read out the global min-max of the column and save them.
		double min = aFDB.min( featureIndex );
		double max = aFDB.max( featureIndex );
		mMinimums.insert( QString::number( featureIndex ), min );
		mMaximums.insert( QString::number( featureIndex ), max );
		
		// Generate kernel densities.
		for ( int labelOutcomeIndex = 0; labelOutcomeIndex < mLabelGroups.size(); ++labelOutcomeIndex )  // Go through label outcomes.
		{
			lpmldata::TabularData FDBByLabelGroup = mFilter.subTableByLabelGroup( aFDB, aLDB, aLabelIndex, mLabelGroups.at( labelOutcomeIndex ) );  // Take the table containing only the given outcomes.

			//qDebug() << FDBByLabelGroup.means();
			//qDebug() << FDBByLabelGroup.deviations();
		
			QVariantList column = FDBByLabelGroup.column( featureIndex );
			QString columnName  = FDBByLabelGroup.columnName( featureIndex );

			lpmleval::FeatureKernel* kernel = new lpmleval::FeatureKernel( column, min, max );  // TODO: Here additional settings can be given later on...

			kernelMax = std::max( kernelMax, kernel->maxDensity() );
			mKernelMap.insert( mLabelGroups.at( labelOutcomeIndex ) + "/" + QString::number( featureIndex ), kernel );
			mKernelMap[ mLabelGroups.at( labelOutcomeIndex ) + "/" + QString::number( featureIndex ) ]->normalize( kernel->maxDensity() );

			mColumnNames.insert( QString::number( featureIndex ), columnName );
		}

		for ( int labelOutcomeIndex = 0; labelOutcomeIndex < mLabelGroups.size(); ++labelOutcomeIndex )  // Go through label outcomes, normalize the kernels by maximum density.
		{
			mKernelMap[ mLabelGroups.at( labelOutcomeIndex ) + "/" + QString::number( featureIndex ) ]->normalize( kernelMax );
		}
	}

	calculateOverlapRatios();
}

//-----------------------------------------------------------------------------

KernelDensityExtractor::~KernelDensityExtractor()
{
	for ( auto kernel : mKernelMap )
	{
		delete kernel;
	}

	mKernelMap.clear();
}

//-----------------------------------------------------------------------------

FeatureKernel* KernelDensityExtractor::kernel( QString aLabelOutcome, int aFeatureIndex )
{
	return mKernelMap.value( aLabelOutcome + "/" + QString::number( aFeatureIndex ) );
}

//-----------------------------------------------------------------------------

FeatureKernel* KernelDensityExtractor::kernel( int aLabelOutcomeIndex, int aFeatureIndex )
{
	return mKernelMap.value( mLabelGroups.at( aLabelOutcomeIndex ) + "/" + QString::number( aFeatureIndex ) );
}

//-----------------------------------------------------------------------------

void KernelDensityExtractor::calculateOverlapRatios()
{
#pragma omp parallel for schedule( dynamic, 1)
	for ( int featureIndex = 0; featureIndex < mFeatureCount; ++featureIndex )  // Go through the kernel densities.
	{
		QSet< QPair< int, int > >  emanVot;

		double area = 0.0;
		double intersection = 0.0;

		for ( int kernelindexA = 0; kernelindexA < mLabelGroups.size(); ++kernelindexA )  // Go through label outcomes.
		{
			FeatureKernel* kernelA = kernel( mLabelGroups.at( kernelindexA ), featureIndex );
			area += areaUnderCurve( kernelA, featureIndex );

			for ( int kernelindexB = 0; kernelindexB < mLabelGroups.size(); ++kernelindexB )  // Go through label outcomes.
			{
				if ( kernelindexA != kernelindexB &&
					!( emanVot.contains( QPair< int, int >( kernelindexA, kernelindexB ) ) || emanVot.contains( QPair< int, int >( kernelindexB, kernelindexA ) ) ) )  // intersect the kernels.
				{
					emanVot.insert( QPair< int, int >( kernelindexA, kernelindexB ) );
					FeatureKernel* kernelB = kernel( mLabelGroups.at( kernelindexB ), featureIndex );
					intersection += intersect( kernelA, kernelB, featureIndex );
				}
			}
		}

#pragma omp critical
		mOverlapRatio.insertMulti( intersection / area, featureIndex );
	}
}

//-----------------------------------------------------------------------------

double KernelDensityExtractor::areaUnderCurve( const FeatureKernel* aFeatureKernel, int aFeatureIndex )
{
	double AUC = 0.0;

	double min = minimum( aFeatureIndex );
	double max = maximum( aFeatureIndex );

	QVector< double > kernelArray = aFeatureKernel->render( min, max );

	for ( int i = 0; i < kernelArray.size(); ++i )
	{
		AUC += kernelArray.at( i );
	}

	return AUC;
}

//-----------------------------------------------------------------------------

double KernelDensityExtractor::intersect( FeatureKernel* aLeft, FeatureKernel* aRight, int aFeatureIndex )
{
	double intersection = 0.0;

	double min = minimum( aFeatureIndex );
	double max = maximum( aFeatureIndex );

	QVector< double > left = aLeft->render( min, max );
	QVector< double > right = aRight->render( min, max );

	int size = std::min( left.size(), right.size() );

	for ( int i = 0; i < size; ++i )
	{
		intersection += std::min( left.at( i ), right.at( i ) );
	}

	return intersection;
}

//-----------------------------------------------------------------------------

}