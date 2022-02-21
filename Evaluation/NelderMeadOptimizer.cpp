/*!
* \file
* NelderMead class defitition. This file is part of Evaluation module.
*
* \remarks
*
* \authors
* lpapp
*/

#include <Evaluation/NelderMeadOptimizer.h>
#include <QDebug>

namespace lpmleval
{

//-----------------------------------------------------------------------------

NelderMeadOptimizer::NelderMeadOptimizer( lpmleval::AbstractModel* aModel, lpmleval::AbstractAnalytics* aAnalytics, const QVector< double >& aInitialInputs, const QVector< double >& aScales, double aTolerance, lint aMaximumIterationCount, bool aIsNegativeNotAllowed )
:
	AbstractOptimizer( nullptr ),
	mAnalytics( aAnalytics ),
	mInitialParameters( aInitialInputs ),
	mScales( aScales ),
	mTolerance( aTolerance ),
	mMaximumIterationCount( aMaximumIterationCount ),
	mTerminatedIterationCount( 0 ),
	mIsPunish( false ),
	mTerminationCode( TerminationCode::FunctionConverged ),
	mIsNegativeNotAllowed( aIsNegativeNotAllowed ),
	mIsStop( false )
{
	mModel = aModel;
}

//-----------------------------------------------------------------------------

NelderMeadOptimizer::~NelderMeadOptimizer()
{
	mScales.clear();
	mInitialParameters.clear();
	mOptimizedParameters.clear();
}

//-----------------------------------------------------------------------------

double NelderMeadOptimizer::amotry( QList< QVector< double > > &p, QVector< double > &y, QVector< double > &psum, double ihi, double fac )
{
	const double fac1 = ( 1.0 - fac ) / psum.size();
	const double fac2 = fac1 - fac;
	QVector< double > ptry;
	QVector< double > ytry;
	ytry.resize( 1 );

	for ( lint i = 0; i < p[ ihi ].size(); ++i )
	{
		ptry.push_back( ( psum[ i ] * fac1 ) - ( p[ ihi ][ i ] * fac2 ) );
	}

	//modifyParamsByPunishment( ptry );
	if ( testIfNegative( ptry ) )
	{
		ytry[ 0 ] = DBL_MAX;
	}
	else
	{
		mModel->set( ptry );		
		ytry[0] = mAnalytics->evaluate( mModel );
		//mFunction->execute( ptry, ytry );
	}

	if ( ytry[ 0 ] < y[ ihi ] )
	{
		y[ ihi ] = ytry[ 0 ];

		for ( ulint i = 0; i < psum.size(); ++i )
		{
			psum[ i ] = psum[ i ] + ptry[ i ] - p[ ihi ][ i ];
		}

		p[ ihi ] = ptry;
	}

	return ytry[ 0 ];
}

//-----------------------------------------------------------------------------

bool NelderMeadOptimizer::testIfNegative( QVector< double > aVector )
{
	if ( mIsNegativeNotAllowed )
	{
		for ( int i = 0; i < aVector.size(); ++i )
		{
			if ( aVector.at( i ) < 0.0 ) return true;
		}
	}

	return false;
}

//-----------------------------------------------------------------------------

void NelderMeadOptimizer::build()
{
	QList< QVector< double > > p;

	for ( int i = 0; i < mModel->inputCount() + 1; ++i )
	{
		p.push_back( mInitialParameters );
	}

	for ( int i = 0; i < mModel->inputCount(); ++i )
	{
		p[ i + 1 ][ i ] = mInitialParameters[ i ] + mScales[ i ];
	}

	const int mpts = mModel->inputCount() + 1;

	QVector< double > y;
	QVector< double > trial;
	trial.resize( 1 );

	for ( int i = 0; i < mpts; ++i )
	{
		if ( mIsStop )
		{
			mIsStop = false;
			mOptimizedParameters = mInitialParameters;

			mTerminationCode = TerminationCode::ExecutionAborted;
			return;
		}

		if ( testIfNegative( p[ i ] ) )
		{
			trial[ 0 ] = DBL_MAX;
		}
		else
		{
			mModel->set( p[ i ] );
			trial[ 0 ] = mAnalytics->evaluate( mModel );
		}

		y.push_back( trial[ 0 ] );
	}

	lint ncalls = 0;

	QVector< double > psum;

	for ( int i = 0; i < mModel->inputCount(); ++i )
	{
		double sum = 0.0;

		for ( int j = 0; j < mpts; ++j )
		{
			sum += p[ j ][ i ];
		}

		psum.push_back( sum );
	}

	int ilo = 0;
	double ihi;
	double inhi;

	mOptimizedParameters.clear();

	while ( ncalls < mMaximumIterationCount )
	{
		QVector< int > s = labelOrder( y );

		ilo = s[ 0 ];
		ihi = s[ mModel->inputCount() ];
		inhi = s[ mModel->inputCount() - 1 ];

		double d = fabs( y[ ihi ] ) + fabs( y[ ilo ] );

		double rtol;

		if ( d != 0.0 )
		{
			rtol = 3.0 * ( fabs( y[ ihi ] - y[ ilo ] ) ) / d;
		}
		else
		{
			rtol = mTolerance / 2.0;
		}

		if ( rtol < mTolerance || mIsStop )
		{
			double t = y[ 0 ];
			y[ 0 ] = y[ ilo ];
			y[ ilo ] = t;
			mTerminatedIterationCount = ncalls;
			mOptimizedParameters = p[ ilo ];
			p[ ilo ] = p[ 0 ];
			p[ 0 ] = mOptimizedParameters;
			mIsStop = false;

			mTerminationCode = TerminationCode::MinFunctionToleranceChangeReached;
			//qDebug() << "NelderMeadOptimizer ITERATIONS (TerminationCode::MinFunctionToleranceChangeReached): " << ncalls;
			return;
		}

		ncalls = ncalls + 2;

		double ytry = amotry( p, y, psum, ihi, -1.0 );

		if ( ytry <= y[ ilo ] )
		{
			ytry = amotry( p, y, psum, ihi, 2.0 );
		}
		else if ( ytry >= y[ inhi ] )
		{
			double ysave = y[ ihi ];
			ytry = amotry( p, y, psum, ihi, 0.5 );

			if ( ytry >= ysave )
			{
				for ( int i = 0; i < mpts; ++i )
				{
					if ( i != ilo )
					{
						psum.clear();

						for ( int j = 0; j < mModel->inputCount(); ++j )
						{
							psum.push_back( 0.5 * ( p[ i ][ j ] + p[ ilo ][ j ] ) );
						}

						p[ i ] = psum;

						//modifyParamsByPunishment( psum );
						if ( testIfNegative( psum ) )
						{
							trial[ 0 ] = DBL_MAX;
						}
						else
						{
							//mFunction->execute( psum, trial );
							mModel->set( psum );
							trial[ 0 ] = mAnalytics->evaluate( mModel );
						}
						y[ i ] = trial[ 0 ];
					}
				}

				ncalls = ncalls + mModel->inputCount();

				psum.clear();

				for ( int i = 0; i < mModel->inputCount(); ++i )
				{
					double sum = 0.0;

					for ( int j = 0; j < mpts; ++j )
					{
						sum += p[ j ][ i ];
					}

					psum.push_back( sum );
				}
			}
		}
		else
		{
			ncalls = ncalls - 1;
		}
	}

	double t = y[ 0 ];
	y[ 0 ] = y[ ilo ];
	y[ ilo ] = t;
	mTerminatedIterationCount = ncalls;
	mOptimizedParameters = p[ ilo ];
	p[ ilo ] = p[ 0 ];
	p[ 0 ] = mOptimizedParameters;

	mTerminationCode = TerminationCode::MaxIterationsReached;
	//qDebug() << "NelderMeadOptimizer ITERATIONS (TerminationCode::MaxIterationsReached): " << ncalls;
	return;
}


//-----------------------------------------------------------------------------

QVector< int > NelderMeadOptimizer::labelOrder( QVector< double > aVect )
{
	QVector< int > order;

	order.clear();
	order.resize( aVect.size() );

	double locmin;
	int minpos;
	for ( int cnt = 0; cnt < aVect.size(); ++cnt )
	{
		minpos = -1;
		locmin = std::numeric_limits< double >::max();

		for ( ulint i = 0; i < aVect.size(); ++i )
		{
			if ( aVect[ i ] < locmin )
			{
				locmin = aVect[ i ];
				minpos = int( i );
			}
		}

		if ( minpos != -1 )
		{
			aVect[ minpos ] = std::numeric_limits< double >::max();
			order[ cnt ] = minpos;
		}
	}

	return order;
}

//-----------------------------------------------------------------------------

}
