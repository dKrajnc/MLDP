/*!
* \file
* AbstractTBPAction class defitition. This file is part of Evaluation module.
* The AbstractTBPAction is an abstract class for describing tabular data manipulation algorithms such as: dimensionality reduction, outlier detection, feature selection, oversampling, undersampling.
*
* \remarks
*
* \authors
* dKrajnc
*/

#pragma once

#include <Evaluation/Export.h>
#include <DataRepresentation/DataPackage.h>
#include <QSettings>

namespace dkeval
{

//-----------------------------------------------------------------------------

/*!
* \brief AbstractTBPAction abstract class for tabular data manipuation
*/

class Evaluation_API AbstractTBPAction
{
		
public:

	/*!
	* \brief Constructor
	*/
	AbstractTBPAction( QSettings* aSettings )
	:
		mSettings( aSettings ),
		mIsInitValid( true )
	{		
	}

	/*!
	* \brief Destructor
	*/
	~AbstractTBPAction() {};


	/*!
	* \brief Builds the algorithm based on the input datapackage 
	* \param [in] aDataPackage The package of feature and label data
	*/
	virtual void build( const lpmldata::DataPackage& aDataPackage ) = 0;

	/*!
	* \brief Transforms the datapackage
	* \param [in] aDataPackage The package of feature and label data
	* \return lpmldata::DataPackage the transformed datapackage
	*/
	virtual lpmldata::DataPackage run( const lpmldata::DataPackage& aDataPackage ) = 0;

	/*!
	* \brief Unique class ID
	* \return QString of class ID
	*/
	virtual QString id() = 0;

	/*!
	* \brief Algorithm hyperparameters
	* \return QMap < QString, QVariant > of hyperparameter names and values
	*/
	virtual QMap< QString, QVariant > parameters() = 0;
	
protected:

	QSettings*  mSettings;
	bool        mIsInitValid;

};

//-----------------------------------------------------------------------------

}
