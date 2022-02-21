/*!
* \file
* CMAnalytics class defitition. This file is part of Evaluation module.
* The CMAnalytics class contains basic confusion matrix analytics metrics calculations
*
* \remarks
*
* \authors
* dKrajnc
*/

#pragma once

#include <QDebug>

namespace dkeval
{
//-----------------------------------------------------------------------------

class CMAnalytics
{

public:

	/*!
	* \brief Constructor to build up confussion matrix.
	* \param [in] aCMelements The confussion matrix.
	*/
	CMAnalytics( const QMap< QString, double >& aCMelements )
		:
		mTP( 0.0 ),
		mTN( 0.0 ),
		mFP( 0.0 ),
		mFN( 0.0 )
	{
		for ( auto& key : aCMelements.keys() )
		{
			if ( key == "TP" )
			{
				mTP = aCMelements.value( key );
			}
			else if ( key == "TN" )
			{
				mTN = aCMelements.value( key );
			}
			else if ( key == "FP" )
			{
				mFP = aCMelements.value( key );
			}
			else if ( key == "FN" )
			{
				mFN = aCMelements.value( key );
			}
		}		
	}

	/*!
	* \brief Destructor
	*/
	~CMAnalytics() {};

public:

	/*!
	* \brief Accuracy calculation based on the confussion matrix
	* \return double value of calculated accuracy
	*/
	double acc() { return ( mTP + mTN ) / ( mTP + mTN + mFP + mFN ); };

	/*!
	* \brief Specificity calculation based on the confussion matrix
	* \return double value of calculated specificity
	*/
	double spc() { return mTN / ( mTN + mFP ); };

	/*!
	* \brief Sensitivity calculation based on the confussion matrix
	* \return double value of calculated sensitivity
	*/
	double sns() { return mTP / ( mTP + mFN ); };

	/*!
	* \brief Negative predictive value calculation based on the confussion matrix
	* \return double value of calculated negative predictive value
	*/
	double npv() { return mTN / ( mTN + mFN ); };

	/*!
	* \brief Positive predictive value calculation based on the confussion matrix
	* \return double value of calculated positive predictive value
	*/
	double ppv() { return mTP / ( mTP + mFP ); };


	/*!
	* \brief Saves the performance at provided location
	* \param [in] aPath The path to the save location
	* \param [in] aFileName The name of the saved file
	*/
	void savePerformanceAt( QString aPath, QString aFileName )
	{
		auto filePath = aPath.append( aFileName );
		QFile file( filePath );
		if ( file.open( QIODevice::WriteOnly | QIODevice::Text ) )
		{			
			QTextStream stream( &file );		
			

			stream << "File path: ;" << filePath << endl;
			stream << endl;
			stream << endl;
			stream << endl;
			stream << ";" << ";" << "TP;" << "TN;" << "FP;" << "FN;" << endl;
			stream << ";" << ";" << mTP << ";" << mTN << ";" << mFP << ";" << mFN << ";" << "\n";
			stream << endl;
			stream << "ACC;" << acc() << endl;;
			stream << "SNS;" << sns() << endl;;
			stream << "SPC;" << spc() << endl;;
			stream << "NPV;" << npv() << endl;;
			stream << "PPV;" << ppv() << endl;;
			stream << endl;;


			file.close();
		}
		else
		{
			qDebug() << "Error, no results saved!";
		}
	}

private:

	double mTP;
	double mTN;
	double mFP;
	double mFN;
};

//-----------------------------------------------------------------------------
}