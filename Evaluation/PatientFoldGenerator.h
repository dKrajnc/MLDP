/*!
* \file
* PatientFoldGenerator class defitition. This file is part of Evaluation module.
* The PatientFoldGenerator is a class for Monte Carlo cross-validation folds generation
* \remarks
*
* \authors
* dkrajnc
*/

#pragma once

#include <DataRepresentation/DataPackage.h>
#include <Evaluation/Export.h>
#include <QVariant>
#include <QVector>


namespace dkeval
{

typedef QPair<std::shared_ptr< lpmldata::DataPackage >, std::shared_ptr< lpmldata::DataPackage >> Pair;

//-----------------------------------------------------------------------------

class Evaluation_API PatientFoldGenerator
{

public:
	
	/*!
	* \brief Constructor to load the datapackage and minimal subsample count information
	* \param [in] aDataPackage The package of feature and label data
	* \param [in] aMinSubsampleCount The definition of minimal subsample count 
	*/
	PatientFoldGenerator( lpmldata::DataPackage aDataPackage, int aMinSubsampleCount );

	/*!
	* \brief Destructor
	*/
	~PatientFoldGenerator() {};

	/*!
	* \brief  Monte Carlo cross-validation folds generation
	* \param [in] aFoldCount The definition of generated fold count
	*/
	void generate( int aFoldCount );

	/*!
	* \brief Get fold at defined index
	* \param [in] aFoldIndex The fold index
	*/
	Pair fold( int aFoldIndex );

	/*!
	* \brief Tests if training data samples are valid
	* \param [in] aTrainingKeys The training data sample keys
	* \return True if the samples are valid, false otherwise.
	*/
	bool isValidTrainingSet( const QList< QString >& aTrainingKeys );

	/*!
	* \brief Tests if validation data samples are valid
	* \param [in] aValidationKeys The validation data sample keys
	* \return True if the samples are valid, false otherwise.
	*/
	bool isValidValidationSet( const QList< QString >& aValidationKeys );

private:

	QList< QList< QString > >  mValidationPatientHistory;
	lpmldata::DataPackage      mDataPackage;
	int                        mMinSubsampleCount;
	QList< QString >           mPatientNames;
};

//-----------------------------------------------------------------------------

}
