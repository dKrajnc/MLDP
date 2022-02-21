/*!
* \file
* Undersampling class defitition. This file is part of Evaluation module.
* The Undersampling is a class responsible for data purification and sample redundancy reduction based on the algorithms such as: Tomek links, random undersampling, etc.
*
* \remarks
*
* \authors
* dkrajnc
*/

#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractTDPAction.h>
#include <QDebug>
#include <random>
#include <set>

namespace dkeval
{

class Evaluation_API Undersampling: public AbstractTBPAction
{

public:

	/*!
	* \brief Constructor to load settings parameters
	* \param [in] aSettings The settigns file
	*/
	Undersampling( QSettings* aSettings )
		:
		AbstractTBPAction( aSettings ),
		mUndersamplingAmount(),
		mChoosenSamples(),
		mAuto( false ),
		mType(),
		mParameters()
	{		
		if ( mSettings == nullptr )
		{
			qDebug() << "RandomUndersampling - Error: Settings is a nullptr";
			mIsInitValid = false;
		}
		else
		{
			bool isType;
			mType = mSettings->value( "Undersampling/type" ).toString();
			if ( mType == "" )
			{
				qDebug() << "Undersampling - Error: Invalid parameter type";
			}

			mParameters.insert( "Undersampling/type", mType );
		}
	}

	/*!
	* \brief Destructor
	*/
	~Undersampling() {};

	/*!
	* \brief Builds the algorithm based on the input datapackage
	* \param [in] aDataPackage The package of feature and label data
	*/
	void build( const lpmldata::DataPackage& aDataPackage ) override;

		/*!
	* \brief Transforms the datapackage based on selected samples
	* \param [in] aDataPackage The package of feature and label data
	* \return lpmldata::DataPackage the transformed datapackage
	*/
	lpmldata::DataPackage run( const lpmldata::DataPackage& aDataPackage ) override;

	/*!
	* \brief Unique class ID
	* \return QString of class ID
	*/
	QString id() override { return "US"; }

	/*!
	* \brief Algorithm hyperparameters
	* \return QMap < QString, QVariant > of hyperparameter names and values
	*/
	QMap < QString, QVariant > parameters() override { return mParameters; }
	

private:
	void randomUndersampling( const lpmldata::DataPackage& aDataPackage );
	void tomekLinks( const lpmldata::DataPackage& aDataPackage );
	int counterDeterminant( const QVariantList& aFeature, QString aFeatureName, const QVariantList& aMinorityFeature, QString aMinorityName, const QVariantList& aMajorityFeature, QString aMajorityName, const double& aDistance );
	double distance( const QVariantList& aFirst, const QVariantList& aSecond );

private:

	int mUndersamplingAmount;
	QStringList mChoosenSamples;
	bool mAuto;
	QString mType;
	QMap< QString, QVariant > mParameters;
};

}