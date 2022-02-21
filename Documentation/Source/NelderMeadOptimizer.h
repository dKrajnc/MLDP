/*!
* \file
* NelderMeadOptimizer class defitition. This file is part of Evaluation module.
*
* \remarks
*
* \authors
* lpapp
*/

#pragma once

#include <Evaluation/Export.h>
#include <DataRepresentation/Types.h>
#include <Evaluation/AbstractOptimizer.h>
#include <Evaluation/AbstractAnalytics.h>
#include <QVector>
#include <QSettings>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API NelderMeadOptimizer: public lpmleval::AbstractOptimizer
{

public:

	enum class TerminationCode
	{
		ParameterError = 0,       // Input array size mismatch.
		FunctionConverged,        // Function converged to a solution.
		MinFunctionToleranceChangeReached,  // Minimum function tolerance changes reached.
		MaxIterationsReached,     // Maximum number of iterations has been reached.
		ExecutionAborted,
	};

	NelderMeadOptimizer( lpmleval::AbstractModel* aModel, lpmleval::AbstractAnalytics* aAnalytics, const QVector< double >& aInitialInputs, const QVector< double >& aScales, double aTolerance, lint aMaximumIterationCount, bool aIsNegativeNotAllowed );

	NelderMeadOptimizer( QSettings* aSettings );

	virtual ~NelderMeadOptimizer();

	void build() override;

	//const QVector< double >& result() { return mOptimizedParameters; }

	/*!
	* \brief Sets the tolerance of the Nelder-Mead iterations.
	* \param [aTolernace] The tolerance of the Nelder-Mead iterations.
	*/
	void setTolerance( double aTolerance ) { mTolerance = aTolerance; }

	/*!
	* \brief Sets the scale values.
	* \param [in] aScale The scale values.
	*/
	void setScales( const QVector< double > &aScales ) { mScales = aScales; }

	/*!
	* \brief Sets the initial parameters.
	* \param [in] aInit The initial parameters.
	*/
	void setInitialParameters( const QVector< double > &aInitialParameters ) { mInitialParameters = aInitialParameters; }

	/*!
	* \brief Sets the maximum iteration number.
	* \param [in] aIteration The maximum iteration number.
	*/
	void setMaximumIterationCount( int aIterations ) { mMaximumIterationCount = aIterations; }

	/*!
	* \brief Returns with the maximum number of iterations.
	* \return The maximum number of iterations.
	*/
	const lint maximumIterationCount() const { return mMaximumIterationCount; }

	/*!
	* \brief Returns with the tolerance value.
	* \return The tolerance value.
	*/
	const double tolerance() const { return mTolerance; }

	/*!
	* \brief Returns with the scale values.
	* \return The scale values.
	*/
	const QVector< double >& scales() const { return mScales; }

	/*!
	* \brief Returns with the initial parameters.
	* \return The initial parameters.
	*/
	const QVector< double >& initialParameters() const { return mInitialParameters; }

	/*!
	* \brief Returns with the iteration counter in which the Nelder-Mead terminated.
	* \return The the iteration counter in which the Nelder-Mead terminated.
	*/
	const lint terminatedIterationCount() const { return mTerminatedIterationCount; }

	/*!
	* \brief Overwrites the input parameters based on the punishment intervals.
	* \param [in,out] aParams the parameters that need to be modified if over the punishment intervals.
	*/
	//void modifyParamsByPunishment( QVector< double > &aParams );

	TerminationCode terminationCode() const { return mTerminationCode; }

	QVector< double > result() override { return mOptimizedParameters; }

private:

	/*!
	* \brief Orders the output of the labelled regions.
	* \param [in] aVector The labels in the vector.
	* \return The ordered vector.
	*/
	QVector< int > labelOrder( QVector< double > aVector );

	/*!
	* \brief Performs one Nelder-Mead iteration.
	* \param [in] p Parameter vector for the iteration.
	* \param [in] y Similarity vector.
	* \param [in] psum New sum value calculated based on the newly generated parameters.
	* \param [in] ihi Highest similarity position.
	* \param [in] fac Defines how the new calculation should be executed.
	* \return The new similarity value based on the new iteration.
	*/
	double amotry( QList< QVector< double > > &p, QVector< double > &y, QVector< double > &psum, double ihi, double fac );

	bool testIfNegative( QVector< double > aVector );

private:

	double                       mTolerance;                 //!< Tolerance of the algorythm
	lint                         mMaximumIterationCount;     //!< Exected iteration number
	QVector< double >            mScales;                    //!< Scale values algorythm
	QVector< double >            mInitialParameters;         //!< Initial parameter vector
	QVector< double >            mOptimizedParameters;       //!< Result of the optimization method.
	lint                         mTerminatedIterationCount;  //!< The last iteration
	bool                         mIsPunish;                  //!< True if the parameters have to be punished.
	TerminationCode              mTerminationCode;           //!< Error code.
	bool                         mIsNegativeNotAllowed;      //!< Are negative values allowed?
	bool                         mIsStop;
	lpmleval::AbstractAnalytics* mAnalytics;

};

//-----------------------------------------------------------------------------

}
